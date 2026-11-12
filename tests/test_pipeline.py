from argparse import Namespace
import json
from unittest.mock import Mock

import pandas as pd
import pytest

import config
import main
from src.matching import build_match_report, eligible_players
from src.rankings import add_rankings
from src.competitiveness import (
    build_roster_analysis,
    build_roster_ownership,
    build_weekly_analysis,
)
from src.representation import (
    build_positional_distributions,
    export_positional_distributions,
)
from src.scoring import score_with_sleeper_settings, scoring_settings_summary
from src.sleeper_api import (
    get_week_matchups,
    normalize_players,
    player_position_audit,
)
from src.stats import (
    fetch_season_stats,
    fetch_weekly_stats,
    stats_to_frame,
    validate_stats_response,
)
from src.visualizations import (
    create_competitiveness_charts,
    create_positional_charts,
)


def test_comparison_scoring_for_qb_rb_wr_and_te():
    players = pd.DataFrame({
        "position": ["RB", "WR", "QB", "TE"],
        "rec": [4, 3, 2, 4],
        "bonus_rec_te": [0, 0, 0, 4],
        "pass_yd": [0, 0, 200, 0],
        "rush_yd": [20, 0, 0, 0],
    })
    settings = {"rec": 1.0, "bonus_rec_te": 1.0, "pass_yd": 0.04, "rush_yd": 0.1}

    current = score_with_sleeper_settings(players, settings)
    comparison = score_with_sleeper_settings(
        players,
        settings,
        reception_points=0.5,
        te_premium=0.5,
        qb_reception_points=1.0,
    )

    assert current.tolist() == [6.0, 3.0, 10.0, 8.0]
    assert comparison.tolist() == [4.0, 1.5, 10.0, 4.0]


def test_comparison_reception_points_exclude_kickers_and_defenses():
    players = pd.DataFrame({
        "position": ["RB", "WR", "TE", "K", "DEF/DST"],
        "rec": [1, 2, 3, 4, 5],
        "rush_yd": [10, 0, 0, 0, 0],
        "pts_allow_0": [0, 0, 0, 0, 1],
    })
    settings = {
        "rec": 1.0,
        "rush_yd": 0.1,
        "pts_allow_0": 10.0,
    }

    current = score_with_sleeper_settings(players, settings)
    comparison = score_with_sleeper_settings(
        players,
        settings,
        reception_points=0.5,
        te_premium=0.5,
        qb_reception_points=1.0,
    )

    assert current.tolist() == [2.0, 2.0, 3.0, 4.0, 15.0]
    assert comparison.tolist() == [1.5, 1.0, 3.0, 0.0, 10.0]
    assert comparison.iloc[3] == current.iloc[3] - 4.0
    assert comparison.iloc[4] == current.iloc[4] - 5.0


def test_current_league_scoring_applies_kicker_and_defense_settings():
    players = pd.DataFrame({
        "position": ["K", "DEF/DST"],
        "fgm_40_49": [2, 0],
        "xpm": [3, 0],
        "sack": [0, 4],
        "int": [0, 2],
        "pts_allow_14_20": [0, 1],
    })
    settings = {
        "fgm_40_49": 5.0,
        "xpm": 1.0,
        "sack": 1.0,
        "int": 2.0,
        "pts_allow_14_20": 1.0,
    }

    assert score_with_sleeper_settings(players, settings).tolist() == [13.0, 9.0]
    assert score_with_sleeper_settings(
        players,
        settings,
        reception_points=0.5,
        te_premium=0.5,
        qb_reception_points=1.0,
    ).tolist() == [13.0, 9.0]


def test_current_uses_actual_sleeper_te_reception_setting():
    player = pd.DataFrame({
        "position": ["TE"],
        "rec": [3],
        "bonus_rec_te": [3],
        "rec_yd": [40],
    })
    settings = {"rec": 1.25, "bonus_rec_te": 0.75, "rec_yd": 0.1}
    assert score_with_sleeper_settings(player, settings).iloc[0] == 10.0


def test_comparison_replaces_base_and_te_reception_values_only():
    player = pd.DataFrame({
        "position": ["TE"],
        "rec": [3],
        "bonus_rec_te": [3],
        "rec_yd": [40],
        "rec_td": [1],
    })
    settings = {
        "rec": 1.0,
        "bonus_rec_te": 1.0,
        "rec_yd": 0.1,
        "rec_td": 6.0,
    }
    current = score_with_sleeper_settings(player, settings).iloc[0]
    comparison = score_with_sleeper_settings(
        player,
        settings,
        reception_points=0.5,
        te_premium=0.5,
        qb_reception_points=1.0,
    ).iloc[0]
    assert current == 16.0
    assert comparison == 13.0


def test_comparison_te_premium_falls_back_to_receptions():
    player = pd.DataFrame({
        "position": ["TE", "TE"],
        "rec": [2, 1],
        "bonus_rec_te": [2, pd.NA],
    })
    points = score_with_sleeper_settings(
        player,
        {"rec": 1},
        reception_points=0.5,
        te_premium=0.5,
        qb_reception_points=1.0,
    )
    assert points.tolist() == [2.0, 1.0]


def test_current_te_premium_falls_back_when_raw_bonus_is_missing():
    player = pd.DataFrame({
        "position": ["TE"],
        "rec": [3],
        "bonus_rec_te": [pd.NA],
    })
    points = score_with_sleeper_settings(
        player, {"rec": 1.0, "bonus_rec_te": 1.0}
    )
    assert points.iloc[0] == 6.0


def test_precomputed_sleeper_point_totals_are_never_used():
    player = pd.DataFrame({
        "position": ["RB"],
        "rec": [2],
        "rush_yd": [50],
        "rush_td": [1],
        "pts_std": [9999],
        "pts_half_ppr": [8888],
        "pts_ppr": [7777],
    })
    assert score_with_sleeper_settings(
        player, {"rec": 1, "rush_yd": 0.1, "rush_td": 6}
    ).iloc[0] == 13.0


def test_scoring_audit_marks_missing_raw_stats_unsupported():
    audit = scoring_settings_summary(
        {
            "rec": 1.0,
            "bonus_rec_te": 1.0,
            "pts_allow_35p": -4.0,
            "sack": 1.0,
            "zero_unknown_key": 0.0,
            "unknown_custom_key": 6.0,
        },
        {"rec", "bonus_rec_te", "sack"},
    ).set_index("setting")
    assert audit.loc["rec", "status"] == "applied"
    assert audit.loc["bonus_rec_te", "status"] == "applied"
    assert audit.loc["pts_allow_35p", "status"] == "unsupported"
    assert audit.loc["sack", "status"] == "applied"
    assert audit.loc["zero_unknown_key", "status"] == "not relevant (zero scoring value)"
    assert audit.loc["unknown_custom_key", "status"] == "unsupported"


def test_rank_movement_sign_and_deterministic_tie_break():
    players = pd.DataFrame({
        "sleeper_id": ["a", "b", "d", "c"],
        "full_name": ["Alpha", "Beta", "Same", "Same"],
        "position": ["WR", "WR", "RB", "RB"],
        "rec": [1, 2, 1, 1],
        "current_points": [10, 5, 10, 10],
        "comparison_points": [5, 10, 10, 10],
    })
    ranked = add_rankings(players).set_index("sleeper_id")
    assert ranked.loc["a", "rank_movement"] == -1
    assert ranked.loc["b", "rank_movement"] == 1
    assert ranked.loc["c", "current_rank"] == 1
    assert ranked.loc["d", "current_rank"] == 2


def _overall_rank_fixture(count=120):
    rows = []
    positions = ("QB", "RB", "WR", "TE", "K", "DEF/DST")
    for index in range(count):
        rows.append({
            "sleeper_id": f"player-{index:03}",
            "full_name": f"Player {index:03}",
            "position": positions[index % len(positions)],
            "rec": index % 8,
            "current_points": float(count - index),
            "comparison_points": float(index + 1),
        })
    return add_rankings(pd.DataFrame(rows))


def test_overall_tiers_have_exact_unique_players_across_positions():
    ranked = _overall_rank_fixture()
    current, comparison, _, _, _ = build_positional_distributions(ranked)

    for table in (current, comparison):
        for _, row in table.iterrows():
            tier_size = int(row["Tier"].replace("Top ", ""))
            assert row["Total"] == tier_size
            assert sum(
                row[position]
                for position in ("QB", "RB", "WR", "TE", "K", "DEF/DST")
            ) == tier_size

    assert ranked["sleeper_id"].nunique() == len(ranked)
    assert ranked["current_overall_rank"].nunique() == len(ranked)
    assert ranked["comparison_overall_rank"].nunique() == len(ranked)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("sleeper_id", None, "missing Sleeper ID"),
        ("position", None, "unsupported position"),
        ("current_overall_rank", 2, "invalid or duplicate current_overall_rank"),
    ],
)
def test_distribution_validation_rejects_bad_ids_positions_and_ranks(
    column, value, message
):
    ranked = _overall_rank_fixture()
    ranked.loc[ranked.index[0], column] = value
    with pytest.raises(ValueError, match=message):
        build_positional_distributions(ranked)


def test_positional_changes_and_percentages_use_overall_tier_counts():
    ranked = _overall_rank_fixture()
    current, comparison, changes, percentages, membership = (
        build_positional_distributions(ranked)
    )
    for _, row in changes.iterrows():
        for position in ("QB", "RB", "WR", "TE", "K", "DEF/DST"):
            assert row[f"{position} count change"] == (
                row[f"{position} comparison count"] - row[f"{position} current count"]
            )
            assert row[f"{position} percentage point change"] == pytest.approx(
                row[f"{position} comparison percentage"]
                - row[f"{position} current percentage"]
            )
        assert row["Total change"] == 0

    percentage_sums = percentages.groupby(["Tier", "Scoring format"])[
        "Percentage"
    ].sum()
    assert all(abs(value - 100.0) < 1e-9 for value in percentage_sums)
    assert set(current["Tier"]) == {
        "Top 10", "Top 20", "Top 50", "Top 75", "Top 100"
    }
    assert set(comparison["Tier"]) == set(current["Tier"])

    top10_changes = membership[membership["Tier"].eq("Top 10")]
    assert len(top10_changes) == 20
    assert set(top10_changes["Membership change"]) == {"entering", "exiting"}
    assert "retained" in set(membership["Membership change"])
    assert {"Position", "Current overall rank", "Comparison overall rank",
            "Current points", "Comparison points"}.issubset(top10_changes.columns)


def test_distribution_exports_and_charts_are_generated_from_summaries(tmp_path):
    ranked = _overall_rank_fixture()
    current, comparison, changes, percentages, _ = export_positional_distributions(
        ranked, tmp_path
    )
    figures = create_positional_charts(
        current, comparison, changes, percentages, tmp_path / "figures"
    )

    expected_csvs = {
        "positional_distribution_current.csv",
        "positional_distribution_half_ppr_te.csv",
        "positional_distribution_changes.csv",
        "positional_distribution_percentages.csv",
        "positional_percentage_point_changes.csv",
        "tier_membership_changes.csv",
        "overall_rankings_current.csv",
        "overall_rankings_half_ppr_te.csv",
    }
    assert expected_csvs <= {path.name for path in tmp_path.glob("*.csv")}
    assert {path.name for path in figures} == {
        "positional_composition_current.png",
        "positional_composition_half_ppr_te.png",
        "positional_composition_side_by_side.png",
        "positional_count_changes.png",
        "positional_percentage_distribution.png",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in figures)
    assert len(pd.read_csv(tmp_path / "overall_rankings_current.csv")) == len(ranked)
    assert len(pd.read_csv(tmp_path / "overall_rankings_half_ppr_te.csv")) == len(ranked)


def test_player_directory_normalization_and_active_filter():
    players = normalize_players({
        "101": {
            "position": "QB", "full_name": "Active QB", "active": True,
            "gsis_id": "unused-external-id",
        },
        "102": {"position": "TE", "full_name": "Inactive with stats", "active": False},
        "103": {"position": "WR", "full_name": "Inactive zero", "active": False},
        "104": {"position": "K", "full_name": "Kicker", "active": True},
        "105": {"position": "DEF", "team": "BUF", "first_name": "Buffalo",
                "last_name": "Bills", "active": True},
        "106": {"position": "DST", "team": "NYJ", "full_name": "Jets",
                "active": True},
    })
    assert "gsis_id" not in players.columns
    assert players.set_index("sleeper_id").loc["105", "position"] == "DEF/DST"
    assert players.set_index("sleeper_id").loc["106", "position"] == "DEF/DST"
    eligible = eligible_players(players, {"102"})
    assert set(eligible["sleeper_id"]) == {"101", "102", "104", "105", "106"}


def test_player_directory_rejects_missing_or_duplicate_normalized_ids():
    with pytest.raises(ValueError, match="without a player ID"):
        normalize_players({None: {"position": "QB", "active": True}})
    with pytest.raises(ValueError, match="duplicate IDs"):
        normalize_players({
            101: {"position": "QB", "active": True},
            "101": {"position": "RB", "active": True},
        })


def test_player_directory_rejects_duplicate_team_defenses():
    with pytest.raises(ValueError, match="multiple DEF/DST entries"):
        normalize_players({
            "BUF": {"position": "DEF", "team": "BUF", "active": True},
            "BUF-DST": {"position": "DST", "team": "BUF", "active": True},
        })


def test_player_directory_audit_reports_included_and_excluded_values():
    audit = player_position_audit({
        "1": {"position": "QB"},
        "2": {"position": "DEF"},
        "3": {"position": "LB"},
        "4": {"position": None},
    }).set_index("Sleeper directory position")
    assert audit.loc["QB", "Included"]
    assert audit.loc["DEF", "Normalized position"] == "DEF/DST"
    assert not audit.loc["LB", "Included"]
    assert "<missing>" in audit.index


def test_match_report_uses_sleeper_ids_and_reports_unmatched_stats_ids():
    players = normalize_players({
        "101": {"position": "QB", "full_name": "Has stats", "active": True},
        "102": {"position": "RB", "full_name": "Empty record", "active": True},
        "103": {"position": "WR", "full_name": "No record", "active": True},
    })
    stats = {"101": {"pass_yd": 100}, "102": {}, "TEAM_X": {"pts_allow_0": 10}}
    eligible = eligible_players(players, set(stats))
    report, matched = build_match_report(eligible, stats, {"101", "102", "103"})
    statuses = report.loc[report["record_type"].eq("eligible_player")].set_index(
        "player_id"
    )["status"]

    assert matched == 2
    assert statuses["101"] == "matched_with_stats"
    assert statuses["102"] == "matched_no_stats"
    assert statuses["103"] == "no_stats_record"
    assert "TEAM_X" in set(
        report.loc[report["record_type"].eq("unmatched_stats_id"), "player_id"]
    )


def test_stats_response_validation_and_empty_records():
    payload = {"101": {"pass_yd": 100, "pts_std": 10}, "102": {}}
    assert validate_stats_response(payload) == payload
    frame = stats_to_frame(payload).set_index("sleeper_id")
    assert frame.loc["101", "pass_yd"] == 100
    assert pd.isna(frame.loc["102", "pass_yd"])
    with pytest.raises(ValueError, match="non-empty object"):
        validate_stats_response([])
    with pytest.raises(ValueError, match="must be numeric"):
        validate_stats_response({"101": {"pass_yd": "100"}})


def test_stats_fetch_uses_sleeper_endpoint_and_caches(monkeypatch, tmp_path):
    cache = tmp_path / "sleeper_stats_2026.json"
    calls = []

    def fake_get_json(url):
        calls.append(url)
        return {"101": {"pass_yd": 250}, "102": {}}

    monkeypatch.setattr("src.stats.get_json", fake_get_json)
    first = fetch_season_stats(2026, cache)
    second = fetch_season_stats(2026, cache)

    assert calls == ["https://api.sleeper.app/v1/stats/nfl/regular/2026"]
    assert first == second
    assert cache.exists()


def test_weekly_stats_fetch_validates_and_caches_nonempty_responses(
    monkeypatch, tmp_path
):
    cache = tmp_path / "weekly_stats.json"
    calls = []

    def fake_get_json(url):
        calls.append(url)
        return {"101": {"rec": 2, "pts_ppr": 1000}}

    monkeypatch.setattr("src.stats.get_json", fake_get_json)
    first = fetch_weekly_stats(2026, 3, cache)
    second = fetch_weekly_stats(2026, 3, cache)

    assert first == second
    assert calls == ["https://api.sleeper.app/v1/stats/nfl/regular/2026/3"]
    assert cache.exists()
    assert validate_stats_response({}, allow_empty=True) == {}
    with pytest.raises(ValueError, match="non-empty object"):
        validate_stats_response({})


def test_empty_weekly_stats_are_not_cached_and_matchups_are_cached(
    monkeypatch, tmp_path
):
    empty_stats_cache = tmp_path / "empty_week.json"
    monkeypatch.setattr("src.stats.get_json", lambda _: {})
    assert fetch_weekly_stats(2026, 9, empty_stats_cache) == {}
    assert not empty_stats_cache.exists()

    calls = []

    def fake_matchups(url):
        calls.append(url)
        return [{"roster_id": 1, "matchup_id": 1}]

    monkeypatch.setattr("src.sleeper_api.get_json", fake_matchups)
    matchup_cache = tmp_path / "matchups.json"
    first = get_week_matchups("league", 2, matchup_cache)
    second = get_week_matchups("league", 2, matchup_cache)
    assert first == second
    assert calls == ["https://api.sleeper.app/v1/league/league/matchups/2"]
    assert matchup_cache.exists()


def test_roster_ownership_and_top100_use_unique_sleeper_id_mappings():
    players = normalize_players({
        "101": {"position": "WR", "full_name": "Receiver", "active": True},
        "102": {"position": "TE", "full_name": "Tight End", "active": True},
        "103": {"position": "RB", "full_name": "Runner", "active": True},
        "104": {"position": "QB", "full_name": "Quarterback", "active": True},
        "105": {"position": "K", "full_name": "Kicker", "active": True},
    })
    rosters = [
        {"roster_id": 1, "owner_id": "u1", "players": ["101"], "taxi": ["105", "102"]},
        {
            "roster_id": 2, "owner_id": "u2", "players": ["102", "103", "104", "999"],
            "reserve": ["102"],
        },
    ]
    ownership, quality, owner_map, labels = build_roster_ownership(
        rosters,
        players,
        [
            {"user_id": "u1", "display_name": "One"},
            {"user_id": "u2", "display_name": "Two"},
        ],
        snapshot_at="2026-11-04T00:00:00+00:00",
    )
    ownership_status = ownership.set_index("Sleeper ID")["Status"]
    assert ownership_status["102"] == "conflicting_ownership"
    assert ownership_status["999"] == "not_in_player_directory"
    assert set(owner_map) == {"101", "103", "104", "105"}
    assert labels == {"1": "One", "2": "Two"}
    assert quality.set_index("Metric").loc[
        "Players mapped to exactly one team", "Value"
    ] == 4

    stats = {
        "101": {"rec": 10},
        "102": {"rec": 2, "bonus_rec_te": 2},
        "103": {"rush_yd": 100},
        "104": {"pass_yd": 100},
        "105": {"xpm": 1},
    }
    stats_frame = stats_to_frame(stats)
    scored = players.merge(stats_frame, on="sleeper_id", how="left")
    scoring = {
        "rec": 1.0, "bonus_rec_te": 1.0, "rush_yd": 0.1,
        "pass_yd": 0.04, "xpm": 1.0,
    }
    scored["current_points"] = score_with_sleeper_settings(scored, scoring)
    scored["comparison_points"] = score_with_sleeper_settings(
        scored, scoring, reception_points=0.5, te_premium=1.0
    )
    ranked = add_rankings(scored)
    tables = build_roster_analysis(
        players,
        stats_frame,
        ranked,
        rosters,
        [
            {"user_id": "u1", "display_name": "One"},
            {"user_id": "u2", "display_name": "Two"},
        ],
        scoring,
        ["QB", "RB", "WR", "TE", "K", "DEF", "FLEX", "SUPER_FLEX"],
        snapshot_at="2026-11-04T00:00:00+00:00",
    )
    distribution = tables["top100_team_distribution"]
    assert distribution.groupby("Scoring format")["Top 100 player count"].sum().tolist() == [4, 4]
    assert tables["top100_team_concentration"]["HHI (0-1)"].between(0, 1).all()
    assert set(tables["rostered_player_rankings"]["sleeper_id"]) == set(owner_map)
    assert set(tables["positional_scarcity"]["Flexible slots per team (not allocated)"]) == {2}
    assert {"roster_ownership_audit", "team_roster_strength",
            "team_positional_construction"}.issubset(tables)


def test_fixed_lineup_weekly_rescoring_reconstructs_outcomes_and_all_play():
    players = normalize_players({
        "101": {"position": "WR", "full_name": "Receiver", "active": True},
        "102": {"position": "RB", "full_name": "Runner", "active": True},
    })
    matchups = [
        {
            "roster_id": 1, "matchup_id": 1, "players": ["101"],
            "starters": ["101"], "starters_points": [8.0], "points": 8.0,
        },
        {
            "roster_id": 2, "matchup_id": 1, "players": ["102"],
            "starters": ["102"], "starters_points": [6.0], "points": 6.0,
        },
    ]
    tables = build_weekly_analysis(
        [{
            "week": 1,
            "stats": {"101": {"rec": 8}, "102": {"rush_yd": 60}},
            "matchups": matchups,
        }],
        players,
        {"rec": 1.0, "rush_yd": 0.1},
        {"1": "One", "2": "Two"},
        expected_team_count=2,
    )

    matchup = tables["weekly_matchup_rescoring"].iloc[0]
    assert matchup["Current margin (team 1 - team 2)"] == 2.0
    assert matchup["Comparison margin (team 1 - team 2)"] == -2.0
    assert matchup["Outcome changed"]
    outcomes = tables["weekly_team_season_outcomes"].set_index(
        ["Roster ID", "Scoring format"]
    )
    assert outcomes.loc[("1", "Current scoring"), "Record"] == "1-0-0"
    assert outcomes.loc[("1", "QB/TE PPR + RB/WR half-PPR"), "Record"] == "0-1-0"
    assert outcomes.loc[("1", "Current scoring"), "All-play record"] == "1-0-0"
    assert outcomes.loc[("2", "QB/TE PPR + RB/WR half-PPR"), "All-play record"] == "1-0-0"
    assert outcomes.loc[
        ("1", "Current scoring"),
        "Head-to-head minus all-play win percentage",
    ] == 0.0
    assert tables["weekly_data_coverage"]["Included"].tolist() == [True]
    assert tables["weekly_competitiveness_summary"][
        "Outcome flips versus current scoring"
    ].tolist() == [1, 1]


def test_unavailable_weekly_stats_preserve_report_schemas():
    players = normalize_players({
        "101": {"position": "QB", "full_name": "Quarterback", "active": True},
    })
    tables = build_weekly_analysis(
        [{
            "week": 9,
            "stats": {},
            "matchups": [],
            "reason": "Current week is in progress.",
        }],
        players,
        {"pass_yd": 0.04},
        {},
    )
    assert not tables["weekly_data_coverage"].loc[0, "Included"]
    assert tables["weekly_data_coverage"].loc[0, "Reason"] == "Current week is in progress."
    assert "Outcome changed" in tables["weekly_matchup_rescoring"].columns
    assert "Outcome flips versus current scoring" in tables[
        "weekly_competitiveness_summary"
    ].columns
    assert "Head-to-head minus all-play win percentage" in tables[
        "weekly_team_season_outcomes"
    ].columns


def test_no_charts_cli_flag():
    assert main.parse_args(["--no-charts"]).no_charts is True
    assert main.parse_args([]).no_charts is False
    assert main.parse_args(["--refresh-weekly"]).refresh_weekly is True


@pytest.mark.parametrize("league_id", ["", "PASTE_YOUR_LEAGUE_ID_HERE"])
def test_missing_or_unconfigured_league_id_fails_early(monkeypatch, league_id):
    monkeypatch.setattr(config, "LEAGUE_ID", league_id)
    with pytest.raises(ValueError, match="Add your Sleeper league ID"):
        main.run(Namespace(
            season=None, refresh_players=False, refresh_stats=False, no_charts=False
        ))


def test_mocked_end_to_end_exports(monkeypatch, tmp_path, capsys):
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(config, "LEAGUE_ID", "fixture-league")
    monkeypatch.setattr(config, "SEASON_OVERRIDE", "")
    monkeypatch.setattr(config, "RAW_DIR", raw_dir)
    monkeypatch.setattr(config, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(config, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(config, "PLAYER_CACHE", raw_dir / "players.json")
    monkeypatch.setattr(
        config, "STATS_CACHE_TEMPLATE", raw_dir / "sleeper_stats_{season}.json"
    )
    monkeypatch.setattr(config, "POSITIONAL_LIMIT", 100)
    monkeypatch.setattr(config, "OVERALL_LIMIT", 250)
    monkeypatch.setattr(main, "get_league", lambda _: {
        "name": "Fixture league",
        "roster_positions": ["QB", "RB", "WR", "TE", "K", "DEF"],
        "settings": {"start_week": 1, "playoff_week_start": 15},
        "scoring_settings": {
            "rec": 1.0, "bonus_rec_te": 1.0, "rec_yd": 0.1, "pass_td": 4.0,
            "fgm_40_49": 5.0, "xpm": 1.0, "sack": 1.0, "int": 2.0,
            "pts_allow_14_20": 1.0,
        },
    })
    monkeypatch.setattr(main, "get_league_rosters", lambda _: [
        {"roster_id": 1, "owner_id": "u1", "players": ["1", "3", "BUF"]},
        {"roster_id": 2, "owner_id": "u2", "players": ["2", "4", "5"]},
    ])
    monkeypatch.setattr(main, "get_league_users", lambda _: [
        {"user_id": "u1", "display_name": "Team One"},
        {"user_id": "u2", "display_name": "Team Two"},
    ])
    monkeypatch.setattr(main, "get_nfl_state", lambda: {"season": "2026", "week": 2})
    monkeypatch.setattr(main, "get_all_players", Mock(return_value={
        "1": {"position": "TE", "full_name": "Tight End", "active": True},
        "2": {"position": "WR", "full_name": "Wide Receiver", "active": True},
        "3": {"position": "QB", "full_name": "Zero Production QB", "active": True},
        "4": {"position": "RB", "full_name": "Inactive RB", "active": False},
        "5": {"position": "K", "full_name": "Kicker", "active": True},
        "BUF": {
            "position": "DEF", "team": "BUF", "first_name": "Buffalo",
            "last_name": "Bills", "active": True,
        },
    }))
    stats = {
        "1": {"rec": 2, "bonus_rec_te": 2, "rec_yd": 20, "pts_ppr": 1000},
        "2": {"rec": 2, "rec_yd": 20, "pts_ppr": 1000},
        "3": {},
        "4": {},
        "5": {"fgm_40_49": 1, "xpm": 1},
        "BUF": {"sack": 2, "int": 1, "pts_allow_14_20": 1},
        "TEAM_X": {"pts_allow_0": 10},
    }
    monkeypatch.setattr(main, "fetch_season_stats", Mock(return_value=stats))
    monkeypatch.setattr(main, "fetch_weekly_stats", Mock(return_value={
        "1": stats["1"],
        "2": stats["2"],
    }))
    monkeypatch.setattr(main, "get_week_matchups", Mock(return_value=[
        {
            "roster_id": 1, "matchup_id": 1,
            "players": ["1", "3", "BUF"], "starters": ["1"],
            "starters_points": [6.0], "points": 6.0,
        },
        {
            "roster_id": 2, "matchup_id": 1,
            "players": ["2", "4", "5"], "starters": ["2"],
            "starters_points": [4.0], "points": 4.0,
        },
    ]))

    main.run(Namespace(
        season=None, refresh_players=False, refresh_stats=False, no_charts=False
    ))

    expected = {
        "top250_current_scoring.csv",
        "top250_half_ppr_te.csv",
        "positional_top100_comparison.csv",
        "tier_membership_changes.csv",
        "biggest_risers.csv",
        "biggest_fallers.csv",
        "positional_distribution_current.csv",
        "positional_distribution_half_ppr_te.csv",
        "positional_distribution_changes.csv",
        "positional_distribution_percentages.csv",
        "positional_percentage_point_changes.csv",
        "overall_rankings_current.csv",
        "overall_rankings_half_ppr_te.csv",
        "player_match_report.csv",
        "scoring_audit.csv",
        "current_sleeper_scoring_settings.csv",
        "run_metadata.json",
        "eligible_position_counts.csv",
        "player_directory_position_audit.csv",
        "roster_ownership_audit.csv",
        "roster_data_quality.csv",
        "rostered_player_rankings.csv",
        "top100_team_distribution.csv",
        "top100_team_changes.csv",
        "top100_team_concentration.csv",
        "top100_player_membership_changes.csv",
        "team_roster_strength.csv",
        "team_positional_construction.csv",
        "positional_scarcity.csv",
        "weekly_data_coverage.csv",
        "weekly_matchup_rescoring.csv",
        "weekly_team_scores.csv",
        "weekly_missing_starter_stats.csv",
        "weekly_team_season_outcomes.csv",
        "weekly_competitiveness_summary.csv",
        "weekly_position_volatility.csv",
        "weekly_by_week_summary.csv",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["league_name"] == "Fixture league"
    assert metadata["season"] == 2026
    assert metadata["stats_records_retrieved"] == 7
    assert metadata["stats_records_matched"] == 6
    assert metadata["unmatched_stats_player_ids"] == ["TEAM_X"]
    assert metadata["eligible_position_counts"] == {
        "QB": 1, "RB": 0, "WR": 1, "TE": 1, "K": 1, "DEF/DST": 1,
    }
    assert (output_dir / "figures" / "positional_composition_current.png").exists()
    assert (output_dir / "figures" / "positional_composition_half_ppr_te.png").exists()
    assert (output_dir / "figures" / "positional_composition_side_by_side.png").exists()
    assert (output_dir / "figures" / "positional_count_changes.png").exists()
    assert (output_dir / "figures" / "positional_percentage_distribution.png").exists()
    assert (output_dir / "figures" / "top100_players_by_team_current.png").exists()
    assert (output_dir / "figures" / "top100_team_position_composition.png").exists()
    assert (output_dir / "figures" / "team_positional_production.png").exists()
    assert (output_dir / "figures" / "weekly_matchup_margin_distributions.png").exists()
    assert (output_dir / "figures" / "weekly_matchup_margins_by_week.png").exists()
    ranked = pd.read_csv(processed_dir / "all_ranked_players.csv").set_index("full_name")
    assert set(ranked.index) == {
        "Tight End", "Wide Receiver", "Zero Production QB", "Kicker", "Buffalo Bills"
    }
    assert ranked.loc["Tight End", "current_points"] == 6.0
    assert ranked.loc["Tight End", "comparison_points"] == 4.0
    assert ranked.loc["Wide Receiver", "current_points"] == 4.0
    assert ranked.loc["Wide Receiver", "comparison_points"] == 3.0
    assert ranked.loc["Zero Production QB", "current_points"] == 0.0
    assert ranked.loc["Kicker", "current_points"] == 6.0
    assert ranked.loc["Buffalo Bills", "current_points"] == 5.0
    assert ranked.loc["Kicker", "comparison_points"] == 6.0
    assert ranked.loc["Buffalo Bills", "comparison_points"] == 5.0
    weekly_matchups = pd.read_csv(output_dir / "weekly_matchup_rescoring.csv")
    assert len(weekly_matchups) == 1
    assert weekly_matchups.loc[0, "Outcome changed"] == False
    assert len(pd.read_csv(output_dir / "weekly_missing_starter_stats.csv")) == 0
    printed = capsys.readouterr().out
    assert "Sleeper stats records retrieved: 7" in printed
    assert "Stats records matched to the full Sleeper player directory: 6" in printed
    assert "TEAM_X" in printed
