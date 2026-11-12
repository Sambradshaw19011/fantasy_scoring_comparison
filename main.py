import argparse
import json
from datetime import datetime, timezone

import pandas as pd

import config
from src.competitiveness import (
    build_roster_analysis,
    build_weekly_analysis,
    export_analysis_tables,
)
from src.matching import build_match_report, eligible_players
from src.lineups import (
    build_lineup_analysis,
    propose_roster_positions,
)
from src.rankings import add_rankings, make_outputs
from src.representation import export_positional_distributions
from src.scoring import (
    score_with_sleeper_settings,
    scoring_settings_summary,
)
from src.sleeper_api import (
    get_all_players,
    get_league,
    get_league_rosters,
    get_league_users,
    get_week_matchups,
    get_nfl_state,
    normalize_players,
    player_position_audit,
)
from src.stats import fetch_season_stats, fetch_weekly_stats, stats_to_frame
from src.visualizations import (
    create_competitiveness_charts,
    create_lineup_analysis_charts,
    create_positional_charts,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare Sleeper scoring with QB/TE PPR and RB/WR half-PPR."
        )
    )
    parser.add_argument(
        "--refresh-players",
        action="store_true",
        help="Download and replace the cached full Sleeper player directory.",
    )
    parser.add_argument(
        "--refresh-stats",
        action="store_true",
        help="Download and replace the cached Sleeper season statistics.",
    )
    parser.add_argument(
        "--refresh-weekly",
        action="store_true",
        help="Download and replace cached weekly stats and matchup snapshots.",
    )
    parser.add_argument(
        "--season",
        type=int,
        help="NFL season to rank; defaults to SEASON in .env or Sleeper's NFL state.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip generating positional distribution PNG charts.",
    )
    return parser.parse_args(argv)


def run(args=None):
    args = parse_args() if args is None else args
    league_id = config.LEAGUE_ID
    if not league_id or league_id == "PASTE_YOUR_LEAGUE_ID_HERE":
        raise ValueError(
            "Add your Sleeper league ID to the project .env file as "
            "SLEEPER_LEAGUE_ID=..."
        )

    league = get_league(league_id)
    rosters = get_league_rosters(league_id)
    users = get_league_users(league_id)
    state = get_nfl_state()
    configured_season = config.SEASON_OVERRIDE
    season = (
        args.season
        if args.season is not None
        else int(configured_season) if configured_season else int(state["season"])
    )
    scoring = league.get("scoring_settings")
    if not isinstance(scoring, dict) or not scoring:
        raise ValueError("Sleeper returned no usable scoring_settings for this league.")
    if not isinstance(league.get("roster_positions"), list):
        raise ValueError("Sleeper returned no usable roster_positions for this league.")

    for directory in (config.RAW_DIR, config.PROCESSED_DIR, config.OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"League: {league.get('name') or league_id}")
    print(f"Season: {season}")
    print(f"Sleeper NFL state: {state.get('season')} week {state.get('week')}")
    print(f"League scoring categories: {len(scoring)}")

    player_map = get_all_players(
        config.PLAYER_CACHE, refresh=args.refresh_players
    )
    directory_ids = {str(player_id) for player_id in player_map}
    all_players = normalize_players(player_map)
    directory_position_audit = player_position_audit(player_map)
    directory_position_audit.to_csv(
        config.OUTPUT_DIR / "player_directory_position_audit.csv", index=False
    )

    stats = fetch_season_stats(
        season,
        config.STATS_CACHE_TEMPLATE.with_name(
            config.STATS_CACHE_TEMPLATE.name.format(season=season)
        ),
        refresh=args.refresh_stats,
    )
    stats_ids = set(stats)
    nonempty_stats_ids = {player_id for player_id, record in stats.items() if record}
    eligible = eligible_players(all_players, nonempty_stats_ids)
    if eligible.empty:
        raise RuntimeError(
            "No active or season-stat QB/RB/WR/TE/K/DEF players were found."
        )

    position_counts = (
        eligible["position"].value_counts()
        .reindex(config.POSITIONS, fill_value=0)
        .rename_axis("Position")
        .reset_index(name="Eligible entries")
    )
    position_counts.to_csv(
        config.OUTPUT_DIR / "eligible_position_counts.csv", index=False
    )
    report, matched_count = build_match_report(eligible, stats, directory_ids)
    report.to_csv(config.OUTPUT_DIR / "player_match_report.csv", index=False)
    unmatched_ids = sorted(stats_ids - directory_ids)
    eligible_with_stats = len(set(eligible["sleeper_id"]) & nonempty_stats_ids)
    print(f"Sleeper stats records retrieved: {len(stats)}")
    print(
        f"Stats records matched to the full Sleeper player directory: "
        f"{matched_count}"
    )
    print(
        f"Eligible active QB/RB/WR/TE/K/DEF players: {len(eligible)}; "
        f"with non-empty stats: {eligible_with_stats}"
    )
    print("Eligible entries by position:")
    print(position_counts.to_string(index=False))
    excluded_positions = directory_position_audit.loc[
        ~directory_position_audit["Included"], "Sleeper directory position"
    ].tolist()
    print(
        "Excluded Sleeper directory position values: "
        + (", ".join(excluded_positions) if excluded_positions else "none")
    )
    print(f"Unmatched stats player IDs: {len(unmatched_ids)}")
    if unmatched_ids:
        print("  " + ", ".join(unmatched_ids))
    run_metadata = {
        "league_name": league.get("name"),
        "season": season,
        "nfl_state_season": state.get("season"),
        "nfl_state_week": state.get("week"),
        "stats_records_retrieved": len(stats),
        "stats_records_matched": matched_count,
        "eligible_players": len(eligible),
        "eligible_position_counts": dict(zip(
            position_counts["Position"], position_counts["Eligible entries"]
        )),
        "eligible_players_with_stats": eligible_with_stats,
        "unmatched_stats_player_ids": unmatched_ids,
    }
    (config.OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )

    stats_frame = stats_to_frame(stats)
    ranked = eligible.merge(stats_frame, on="sleeper_id", how="left")
    source_stat_keys = set()
    for player_id in eligible["sleeper_id"]:
        source_stat_keys.update(stats.get(player_id, {}).keys())

    ranked["current_points"] = score_with_sleeper_settings(ranked, scoring)
    ranked["comparison_points"] = score_with_sleeper_settings(
        ranked,
        scoring,
        reception_points=config.COMPARISON_BASE_RECEPTION_POINTS,
        te_premium=config.COMPARISON_TE_PREMIUM,
        qb_reception_points=config.COMPARISON_QB_RECEPTION_POINTS,
    )
    ranked = add_rankings(ranked)
    ranked.to_csv(config.PROCESSED_DIR / "all_ranked_players.csv", index=False)

    audit = scoring_settings_summary(scoring, source_stat_keys)
    audit.to_csv(config.OUTPUT_DIR / "scoring_audit.csv", index=False)
    audit[["setting", "points_per_stat"]].to_csv(
        config.OUTPUT_DIR / "current_sleeper_scoring_settings.csv", index=False
    )
    roster_snapshot_at = datetime.now(timezone.utc).isoformat()
    roster_tables = build_roster_analysis(
        all_players,
        stats_frame,
        ranked,
        rosters,
        users,
        scoring,
        league["roster_positions"],
        reception_points=config.COMPARISON_BASE_RECEPTION_POINTS,
        te_premium=config.COMPARISON_TE_PREMIUM,
        qb_reception_points=config.COMPARISON_QB_RECEPTION_POINTS,
        snapshot_at=roster_snapshot_at,
        directory_ids=directory_ids,
    )
    export_analysis_tables(roster_tables, config.OUTPUT_DIR)

    start_week = max(1, int(league.get("settings", {}).get("start_week") or 1))
    state_week = int(state.get("week") or 0)
    regular_season_end = state_week - 1
    playoff_start = int(league.get("settings", {}).get("playoff_week_start") or 0)
    if playoff_start > start_week:
        regular_season_end = min(regular_season_end, playoff_start - 1)
    weekly_data = []
    for week in range(start_week, regular_season_end + 1):
        weekly_stats = fetch_weekly_stats(
            season,
            week,
            config.RAW_DIR / f"sleeper_weekly_stats_{season}_{week}.json",
            refresh=getattr(args, "refresh_weekly", False),
        )
        weekly_matchups = (
            get_week_matchups(
                league_id,
                week,
                config.RAW_DIR / f"sleeper_matchups_{league_id}_{week}.json",
                refresh=getattr(args, "refresh_weekly", False),
            )
            if weekly_stats
            else []
        )
        weekly_data.append({
            "week": week,
            "stats": weekly_stats,
            "matchups": weekly_matchups,
        })
    if state_week >= start_week and state_week > regular_season_end:
        weekly_data.append({
            "week": state_week,
            "stats": {},
            "matchups": [],
            "reason": "Excluded current NFL state week; only completed prior weeks are scored.",
        })
    weekly_tables = build_weekly_analysis(
        weekly_data,
        all_players,
        scoring,
        {
            str(row["Roster ID"]): str(row["Team"])
            for row in roster_tables["top100_team_distribution"]
            .loc[lambda frame: frame["Scoring format"].eq("Current scoring")]
            .to_dict("records")
        },
        reception_points=config.COMPARISON_BASE_RECEPTION_POINTS,
        te_premium=config.COMPARISON_TE_PREMIUM,
        qb_reception_points=config.COMPARISON_QB_RECEPTION_POINTS,
        expected_team_count=len(rosters),
    )
    export_analysis_tables(weekly_tables, config.OUTPUT_DIR)
    lineup_tables = build_lineup_analysis(
        roster_tables["rostered_player_rankings"],
        stats_frame,
        league["roster_positions"],
        weekly_data=weekly_data,
    )
    export_analysis_tables(lineup_tables, config.OUTPUT_DIR)
    proposed_roster_positions = propose_roster_positions(
        league["roster_positions"]
    )
    lineup_settings = {
        "league_name": league.get("name"),
        "season": season,
        "statistical_period": (
            f"Season-to-date through completed Week {regular_season_end}"
        ),
        "roster_snapshot_retrieved_at_utc": roster_snapshot_at,
        "current_roster_positions": league["roster_positions"],
        "proposed_roster_positions": proposed_roster_positions,
    }
    (config.OUTPUT_DIR / "lineup_scenario_settings.json").write_text(
        json.dumps(lineup_settings, indent=2) + "\n", encoding="utf-8"
    )
    included_weeks = weekly_tables["weekly_data_coverage"].loc[
        weekly_tables["weekly_data_coverage"]["Included"], "Week"
    ].astype(int).tolist()
    quality = roster_tables["roster_data_quality"]
    quality.loc[
        quality["Metric"].eq("Historical roster snapshots available by week"),
        "Value",
    ] = ", ".join(map(str, included_weeks)) if included_weeks else "none"
    quality.to_csv(config.OUTPUT_DIR / "roster_data_quality.csv", index=False)
    run_metadata.update({
        "roster_snapshot_retrieved_at_utc": roster_snapshot_at,
        "team_count": len(rosters),
        "rostered_eligible_players_mapped": len(
            roster_tables["rostered_player_rankings"]
        ),
        "validated_historical_matchup_weeks": included_weeks,
        "current_week_excluded": state_week if state_week > regular_season_end else None,
    })
    (config.OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    positional = make_outputs(
        ranked,
        config.OUTPUT_DIR,
        positional_limit=config.POSITIONAL_LIMIT,
        overall_limit=config.OVERALL_LIMIT,
    )
    current_distribution, comparison_distribution, changes, percentages, membership = (
        export_positional_distributions(ranked, config.OUTPUT_DIR)
    )
    if not getattr(args, "no_charts", False):
        figures = create_positional_charts(
            current_distribution,
            comparison_distribution,
            changes,
            percentages,
            config.OUTPUT_DIR / "figures",
        )
        team_figures = create_competitiveness_charts(
            roster_tables,
            weekly_tables,
            config.OUTPUT_DIR / "figures",
        )
        lineup_figures = create_lineup_analysis_charts(
            lineup_tables,
            config.OUTPUT_DIR / "figures",
        )
        print(
            f"Charts written: {', '.join(path.name for path in [*figures, *team_figures, *lineup_figures])}"
        )

    unsupported = audit.loc[audit["status"].eq("unsupported"), "setting"].tolist()
    print(f"Scoring categories unsupported for these player rankings: {len(unsupported)}")
    if unsupported:
        print("  " + ", ".join(unsupported))
    print("\nTop positional movers (positive = rises in comparison format):")
    cols = [
        "full_name", "position", "current_rank", "comparison_rank",
        "rank_movement", "current_points", "comparison_points",
    ]
    print(
        ranked.sort_values(
            ["rank_movement", "position", "comparison_rank"],
            ascending=[False, True, True],
        )[cols].head(15).to_string(index=False)
    )
    print("\nOverall positional representation (comparison - current):")
    print(changes.to_string(index=False))
    print(f"Overall tier player changes: {len(membership)}")
    print(
        f"Roster analysis: {len(rosters)} teams, "
        f"{len(roster_tables['rostered_player_rankings'])} uniquely mapped eligible players"
    )
    print(f"Validated historical matchup weeks: {included_weeks or 'none'}")
    print(f"\nPositional comparison rows exported: {len(positional)}")
    print(f"Outputs written to: {config.OUTPUT_DIR}")


def main():
    args = parse_args()
    try:
        run(args)
    except (RuntimeError, ValueError, OSError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
