from collections import Counter, defaultdict
from collections.abc import KeysView
from datetime import datetime, timezone
import math
from typing import Any

import pandas as pd

from src.rankings import add_rankings
from src.scoring import score_with_sleeper_settings

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF/DST")
POSITION_ALIASES = {"DEF": "DEF/DST", "DST": "DEF/DST", "D/ST": "DEF/DST"}
FORMAT_COLUMNS = (
    ("Current scoring", "current_points", "current_overall_rank", "current_rank"),
    (
        "QB/TE PPR + RB/WR half-PPR",
        "comparison_points",
        "comparison_overall_rank",
        "comparison_rank",
    ),
)
WEEKLY_FORMAT_COLUMNS = (
    ("Current scoring", "Current points"),
    ("QB/TE PPR + RB/WR half-PPR", "Comparison points"),
)
EMPTY_STARTER_IDS = {"", "0", "NONE"}


def _normalized_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _roster_members(roster: dict) -> set[str]:
    members: set[str] = set()
    for field in ("players", "taxi", "reserve"):
        values = roster.get(field) or []
        if isinstance(values, dict):
            values = values.keys()
        if not isinstance(values, (list, tuple, set, KeysView)):
            raise ValueError(f"Sleeper roster field {field!r} must contain player IDs.")
        members.update(
            player_id
            for value in values
            if (player_id := _normalized_id(value))
        )
    return members

def _roster_order(roster_ids: list[str]) -> list[str]:
    def key(roster_id: str) -> tuple[int, int | str]:
        try:
            return (0, int(roster_id))
        except ValueError:
            return (1, roster_id.casefold())

    return sorted(roster_ids, key=key)


def _team_labels(rosters: list[dict], users: list[dict]) -> dict[str, str]:
    user_names = {}
    for user in users:
        user_id = _normalized_id(user.get("user_id"))
        metadata = user.get("metadata") or {}
        label = metadata.get("team_name") or user.get("display_name")
        if user_id and label:
            user_names[user_id] = str(label).strip()

    labels = {}
    for roster in rosters:
        roster_id = _normalized_id(roster.get("roster_id"))
        if not roster_id:
            continue
        metadata = roster.get("metadata") or {}
        owner_id = _normalized_id(roster.get("owner_id"))
        label = metadata.get("team_name") or user_names.get(owner_id)
        labels[roster_id] = str(label).strip() if label else f"Roster {roster_id}"

    duplicate_labels = Counter(label.casefold() for label in labels.values())
    return {
        roster_id: (
            f"{label} (Roster {roster_id})"
            if duplicate_labels[label.casefold()] > 1
            else label
        )
        for roster_id, label in labels.items()
    }


def build_roster_ownership(
    rosters: list[dict],
    players: pd.DataFrame,
    users: list[dict] | None = None,
    snapshot_at: str | None = None,
    directory_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, str]]:
    """Return a player-level ownership audit and compact roster coverage summary."""
    if not isinstance(rosters, list):
        raise ValueError("Sleeper roster response must be a list.")
    roster_by_id: dict[str, dict] = {}
    player_rosters: dict[str, set[str]] = defaultdict(set)
    for roster in rosters:
        if not isinstance(roster, dict):
            raise ValueError("Sleeper roster response contains a non-object record.")
        roster_id = _normalized_id(roster.get("roster_id"))
        if not roster_id:
            raise ValueError("Sleeper roster record is missing roster_id.")
        if roster_id in roster_by_id:
            raise ValueError(f"Sleeper returned duplicate roster_id {roster_id}.")
        roster_by_id[roster_id] = roster
        for player_id in _roster_members(roster):
            player_rosters[player_id].add(roster_id)

    labels = _team_labels(rosters, users or [])
    known_directory_ids = (
        {str(player_id) for player_id in directory_ids}
        if directory_ids is not None
        else set(players["sleeper_id"].astype(str))
    )
    owner_map: dict[str, str] = {}
    audit_rows = []
    for player_id in sorted(player_rosters):
        roster_ids = _roster_order(sorted(player_rosters[player_id]))
        if len(roster_ids) > 1:
            status = "conflicting_ownership"
        elif player_id not in known_directory_ids:
            status = "not_in_player_directory"
        else:
            status = "mapped_to_one_team"
            owner_map[player_id] = roster_ids[0]
        audit_rows.append({
            "Sleeper ID": player_id,
            "Roster IDs": ", ".join(roster_ids),
            "Team names": ", ".join(labels.get(rid, f"Roster {rid}") for rid in roster_ids),
            "Status": status,
        })

    ownership = pd.DataFrame(
        audit_rows,
        columns=["Sleeper ID", "Roster IDs", "Team names", "Status"],
    )
    unique_count = len(player_rosters)
    mapped_count = sum(len(roster_ids) == 1 for roster_ids in player_rosters.values())
    conflicting_count = sum(len(roster_ids) > 1 for roster_ids in player_rosters.values())
    unmapped_count = sum(
        len(roster_ids) == 1 and player_id not in known_directory_ids
        for player_id, roster_ids in player_rosters.items()
    )
    summary = pd.DataFrame([
        ("Team count", len(roster_by_id)),
        ("Unique rostered player IDs", unique_count),
        ("Players mapped to exactly one team", mapped_count - unmapped_count),
        ("Conflicting ownership player IDs", conflicting_count),
        ("Rostered IDs missing from the player directory", unmapped_count),
        ("Current roster snapshot retrieved at (UTC)", snapshot_at or datetime.now(timezone.utc).isoformat()),
        ("Historical roster snapshots available by week", "See weekly data coverage"),
    ], columns=["Metric", "Value"])
    return ownership, summary, owner_map, labels


def _scored_rostered_players(
    players: pd.DataFrame,
    stats_frame: pd.DataFrame,
    owner_map: dict[str, str],
    scoring_settings: dict,
    reception_points: float,
    te_premium: float,
    qb_reception_points: float | None,
) -> pd.DataFrame:
    if stats_frame["sleeper_id"].astype(str).duplicated().any():
        raise ValueError("Season statistics contain duplicate Sleeper player IDs.")
    pool = players.loc[
        players["sleeper_id"].astype(str).isin(owner_map)
        & players["position"].isin(POSITIONS)
    ].copy()
    scored = pool.merge(stats_frame, on="sleeper_id", how="left", validate="one_to_one")
    scored["current_points"] = score_with_sleeper_settings(
        scored, scoring_settings
    )
    scored["comparison_points"] = score_with_sleeper_settings(
        scored,
        scoring_settings,
        reception_points=reception_points,
        te_premium=te_premium,
        qb_reception_points=qb_reception_points,
    )
    scored = add_rankings(scored)
    scored["roster_id"] = scored["sleeper_id"].map(owner_map)
    return scored


def _direct_position_slots(roster_positions: list[str] | None) -> tuple[dict[str, int], int]:
    slots = Counter()
    flexible_slots = 0
    for raw_position in roster_positions or []:
        if not isinstance(raw_position, str):
            continue
        position = raw_position.strip().upper()
        if position in ("FLEX", "SUPER_FLEX"):
            flexible_slots += 1
            continue
        normalized = POSITION_ALIASES.get(position, position)
        if normalized in POSITIONS:
            slots[normalized] += 1
    return {position: slots.get(position, 0) for position in POSITIONS}, flexible_slots


def build_positional_scarcity(
    ranked: pd.DataFrame,
    team_count: int,
    roster_positions: list[str] | None,
) -> pd.DataFrame:
    """Use direct league starter slots; flex slots are reported, not reassigned."""
    slots_per_team, flexible_slots = _direct_position_slots(roster_positions)
    rows = []
    for scoring_format, points_column, _, rank_column in FORMAT_COLUMNS:
        for position in POSITIONS:
            group = ranked.loc[ranked["position"].eq(position)].sort_values(
                rank_column, kind="mergesort"
            )
            points = pd.to_numeric(group[points_column], errors="coerce")
            values = points.tolist()
            starter_rank = slots_per_team[position] * team_count
            replacement_target = starter_rank + team_count if starter_rank else 0
            starter_available = values[starter_rank - 1] if starter_rank and len(values) >= starter_rank else float("nan")
            replacement_available = (
                values[replacement_target - 1]
                if replacement_target and len(values) >= replacement_target
                else float("nan")
            )
            elite = values[0] if values else float("nan")
            rows.append({
                "Scoring format": scoring_format,
                "Position": position,
                "Teams": team_count,
                "Direct starting slots per team": slots_per_team[position],
                "Flexible slots per team (not allocated)": flexible_slots,
                "Starter threshold rank": starter_rank,
                "Replacement threshold rank": replacement_target,
                "Eligible player count": len(values),
                "Starter threshold points": starter_available,
                "Replacement threshold points": replacement_available,
                "Elite-to-starter gap": elite - starter_available if pd.notna(starter_available) and pd.notna(elite) else float("nan"),
                "Starter-to-replacement gap": starter_available - replacement_available if pd.notna(starter_available) and pd.notna(replacement_available) else float("nan"),
                "Elite-to-replacement gap": elite - replacement_available if pd.notna(replacement_available) and pd.notna(elite) else float("nan"),
                "Threshold method": "Direct position slots × teams; replacement is one additional player per team. FLEX/SUPER_FLEX are not allocated.",
            })
    return pd.DataFrame(rows)


def build_roster_analysis(
    players: pd.DataFrame,
    stats_frame: pd.DataFrame,
    ranked: pd.DataFrame,
    rosters: list[dict],
    users: list[dict] | None,
    scoring_settings: dict,
    roster_positions: list[str] | None,
    reception_points: float = 0.5,
    te_premium: float = 0.5,
    qb_reception_points: float | None = 1.0,
    snapshot_at: str | None = None,
    directory_ids: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    ownership, quality, owner_map, labels = build_roster_ownership(
        rosters, players, users, snapshot_at, directory_ids
    )
    roster_ids = _roster_order(sorted({
        _normalized_id(roster.get("roster_id"))
        for roster in rosters
        if _normalized_id(roster.get("roster_id"))
    }))
    rostered_ranked = _scored_rostered_players(
        players, stats_frame, owner_map, scoring_settings,
        reception_points, te_premium, qb_reception_points,
    )
    if rostered_ranked.empty:
        raise ValueError("No directory-mapped eligible rostered players are available.")
    rostered_ranked["Team"] = rostered_ranked["roster_id"].map(labels)

    team_rows = []
    for scoring_format, points_column, rank_column, _ in FORMAT_COLUMNS:
        top = rostered_ranked.loc[rostered_ranked[rank_column] <= 100]
        denominator = len(top)
        for roster_id in roster_ids:
            members = top.loc[top["roster_id"].eq(roster_id)]
            row = {
                "Scoring format": scoring_format,
                "Roster ID": roster_id,
                "Team": labels.get(roster_id, f"Roster {roster_id}"),
                "Top 100 player count": len(members),
                "Top 100 share (%)": len(members) / denominator * 100 if denominator else 0.0,
                "Average Top 100 rank": members[rank_column].mean() if len(members) else float("nan"),
                "Best Top 100 rank": members[rank_column].min() if len(members) else float("nan"),
            }
            row.update({
                f"{position} Top 100 count": int(members["position"].eq(position).sum())
                for position in POSITIONS
            })
            team_rows.append(row)
    distribution = pd.DataFrame(team_rows)

    current = distribution.loc[distribution["Scoring format"].eq("Current scoring")].set_index("Roster ID")
    comparison = distribution.loc[
        distribution["Scoring format"].eq("QB/TE PPR + RB/WR half-PPR")
    ].set_index("Roster ID")
    changes = []
    for roster_id in roster_ids:
        current_row = current.loc[roster_id]
        comparison_row = comparison.loc[roster_id]
        row = {
            "Roster ID": roster_id,
            "Team": current_row["Team"],
            "Current Top 100 count": int(current_row["Top 100 player count"]),
            "Comparison Top 100 count": int(comparison_row["Top 100 player count"]),
            "Top 100 count change": int(
                comparison_row["Top 100 player count"] - current_row["Top 100 player count"]
            ),
            "Current Top 100 share (%)": current_row["Top 100 share (%)"],
            "Comparison Top 100 share (%)": comparison_row["Top 100 share (%)"],
            "Top 100 share change (percentage points)": (
                comparison_row["Top 100 share (%)"] - current_row["Top 100 share (%)"]
            ),
        }
        for position in POSITIONS:
            column = f"{position} Top 100 count"
            row[f"{position} Top 100 count change"] = int(
                comparison_row[column] - current_row[column]
            )
        changes.append(row)
    changes_df = pd.DataFrame(changes)

    concentration_rows = []
    for scoring_format, frame in distribution.groupby("Scoring format", sort=False):
        counts = frame["Top 100 player count"].astype(float)
        sample_size = int(counts.sum())
        shares = counts / sample_size if sample_size else counts
        concentration_rows.append({
            "Scoring format": scoring_format,
            "Top 100 sample size": sample_size,
            "Mean players per team": counts.mean(),
            "Population standard deviation": counts.std(ddof=0),
            "Range (max - min)": counts.max() - counts.min(),
            "Teams with zero Top 100 players": int(counts.eq(0).sum()),
            "Teams represented": int(counts.gt(0).sum()),
            "HHI (0-1)": float((shares ** 2).sum()),
            "HHI definition": "Sum of squared team shares of the rostered-player Top 100; 0-1 scale.",
        })
    concentration = pd.DataFrame(concentration_rows)

    current_ids = set(
        rostered_ranked.loc[rostered_ranked["current_overall_rank"] <= 100, "sleeper_id"]
    )
    comparison_ids = set(
        rostered_ranked.loc[rostered_ranked["comparison_overall_rank"] <= 100, "sleeper_id"]
    )
    by_id = rostered_ranked.set_index("sleeper_id")
    membership_rows = []
    for player_id in sorted(current_ids | comparison_ids):
        player = by_id.loc[player_id]
        current_rank = int(player["current_overall_rank"])
        comparison_rank = int(player["comparison_overall_rank"])
        if player_id not in current_ids:
            status = "entering"
        elif player_id not in comparison_ids:
            status = "exiting"
        elif current_rank != comparison_rank:
            status = "retained_rank_changed"
        else:
            continue
        membership_rows.append({
            "Membership change": status,
            "Sleeper ID": player_id,
            "Player": player["full_name"],
            "Position": player["position"],
            "Roster ID": player["roster_id"],
            "Team": labels.get(player["roster_id"], f"Roster {player['roster_id']}"),
            "Current overall rank": current_rank,
            "Comparison overall rank": comparison_rank,
            "Rank movement": current_rank - comparison_rank,
            "Current points": player["current_points"],
            "Comparison points": player["comparison_points"],
        })
    membership = pd.DataFrame(
        membership_rows,
        columns=[
            "Membership change", "Sleeper ID", "Player", "Position", "Roster ID",
            "Team", "Current overall rank", "Comparison overall rank",
            "Rank movement", "Current points", "Comparison points",
        ],
    )

    scarcity = build_positional_scarcity(ranked, len(roster_ids), roster_positions)
    global_ranks = ranked.set_index("sleeper_id")
    slots_per_team, _ = _direct_position_slots(roster_positions)
    position_rows = []
    strength_rows = []
    for roster_id in roster_ids:
        team_pool = rostered_ranked.loc[rostered_ranked["roster_id"].eq(roster_id)]
        for scoring_format, points_column, _, global_rank_column in FORMAT_COLUMNS:
            team_points = pd.to_numeric(team_pool[points_column], errors="coerce").fillna(0)
            ordered_team_points = team_points.sort_values(ascending=False)
            total_points = float(team_points.sum())
            strength = {
                "Scoring format": scoring_format,
                "Roster ID": roster_id,
                "Team": labels.get(roster_id, f"Roster {roster_id}"),
                "Rostered eligible players": len(team_pool),
                "Season-to-date points on current roster": total_points,
            }
            for limit in (3, 5, 10):
                top_total = float(ordered_team_points.head(limit).sum())
                strength[f"Top {limit} player points"] = top_total
                strength[f"Top {limit} average points"] = (
                    top_total / min(limit, len(ordered_team_points))
                    if len(ordered_team_points)
                    else float("nan")
                )
            strength_rows.append(strength)

            for position in POSITIONS:
                position_pool = team_pool.loc[team_pool["position"].eq(position)].copy()
                position_total = float(
                    pd.to_numeric(position_pool[points_column], errors="coerce")
                    .fillna(0).sum()
                )
                direct_slots = slots_per_team[position]
                top_starters = pd.to_numeric(
                    position_pool[points_column], errors="coerce"
                ).fillna(0).sort_values(ascending=False).head(direct_slots)
                rank_limit = global_ranks[global_rank_column].reindex(
                    position_pool["sleeper_id"].astype(str)
                )
                threshold_rows = scarcity.loc[
                    scarcity["Scoring format"].eq(scoring_format)
                    & scarcity["Position"].eq(position)
                ]
                starter_limit = int(threshold_rows.iloc[0]["Starter threshold rank"])
                replacement_limit = int(threshold_rows.iloc[0]["Replacement threshold rank"])
                rank_values = rank_limit.to_numpy()
                position_points = pd.to_numeric(
                    position_pool[points_column], errors="coerce"
                ).fillna(0).to_numpy()
                top_scorer = float(position_points.max()) if len(position_points) else float("nan")
                position_rows.append({
                    "Scoring format": scoring_format,
                    "Roster ID": roster_id,
                    "Team": labels.get(roster_id, f"Roster {roster_id}"),
                    "Position": position,
                    "Rostered player count": len(position_pool),
                    "Season-to-date points on current roster": position_total,
                    "Share of team's position-summed points (%)": (
                        position_total / total_points * 100 if total_points > 0 else float("nan")
                    ),
                    "Top positional scorer points": top_scorer,
                    "Points outside direct positional starters": position_total - float(top_starters.sum()),
                    "Rostered players within starter threshold": int(
                        pd.Series(rank_values).le(starter_limit).sum()
                    ) if starter_limit else 0,
                    "Rostered players within replacement threshold": int(
                        pd.Series(rank_values).le(replacement_limit).sum()
                    ) if replacement_limit else 0,
                })

    return {
        "roster_ownership_audit": ownership,
        "roster_data_quality": quality,
        "rostered_player_rankings": rostered_ranked,
        "top100_team_distribution": distribution,
        "top100_team_changes": changes_df,
        "top100_team_concentration": concentration,
        "top100_player_membership_changes": membership,
        "team_roster_strength": pd.DataFrame(strength_rows),
        "team_positional_construction": pd.DataFrame(position_rows),
        "positional_scarcity": scarcity,
    }


def _number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Sleeper matchup {label} must be numeric, got {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"Sleeper matchup {label} must be finite, got {value!r}.")
    return number


def _valid_player_ids(values: Any, field: str, week: int, roster_id: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(
            f"Sleeper week {week} roster {roster_id} has no list for {field}."
        )
    return [_normalized_id(value) for value in values]


def _score_week(
    week_stats: dict[str, dict],
    players: pd.DataFrame,
    scoring_settings: dict,
    reception_points: float,
    te_premium: float,
    qb_reception_points: float | None,
) -> tuple[dict[str, float], dict[str, float]]:
    if not isinstance(week_stats, dict) or not week_stats:
        raise ValueError("Weekly scoring requires a non-empty Sleeper stats object.")
    stats_frame = pd.DataFrame.from_dict(week_stats, orient="index")
    stats_frame.index = stats_frame.index.astype(str)
    stats_frame.index.name = "sleeper_id"
    stats_frame = stats_frame.reset_index()
    player_positions = players[["sleeper_id", "position"]].drop_duplicates("sleeper_id")
    if players["sleeper_id"].astype(str).duplicated().any():
        raise ValueError("Sleeper player directory has duplicate IDs in weekly scoring.")
    joined = stats_frame.merge(player_positions, on="sleeper_id", how="left", validate="one_to_one")
    current = score_with_sleeper_settings(joined, scoring_settings)
    comparison = score_with_sleeper_settings(
        joined,
        scoring_settings,
        reception_points=reception_points,
        te_premium=te_premium,
        qb_reception_points=qb_reception_points,
    )
    return (
        dict(zip(joined["sleeper_id"].astype(str), current.astype(float))),
        dict(zip(joined["sleeper_id"].astype(str), comparison.astype(float))),
    )


def build_weekly_analysis(
    weeks: list[dict],
    players: pd.DataFrame,
    scoring_settings: dict,
    roster_labels: dict[str, str],
    reception_points: float = 0.5,
    te_premium: float = 0.5,
    qb_reception_points: float | None = 1.0,
    expected_team_count: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Re-score verified Sleeper weekly lineups, retaining their actual starters."""
    coverage_rows = []
    matchup_rows = []
    team_week_rows = []
    player_week_rows = []
    missing_starter_rows = []
    for data in sorted(weeks, key=lambda item: int(item["week"])):
        week = int(data["week"])
        stats = data.get("stats")
        matchups = data.get("matchups")
        if not isinstance(stats, dict) or not stats:
            coverage_rows.append({
                "Week": week,
                "Stats records": len(stats) if isinstance(stats, dict) else 0,
                "Matchup team records": len(matchups) if isinstance(matchups, list) else 0,
                "Starter records": 0,
                "Starters without stat records": 0,
                "Starters missing directory mapping": 0,
                "Maximum current-score reconciliation error": float("nan"),
                "Included": False,
                "Reason": data.get(
                    "reason", "Sleeper weekly statistics are not available."
                ),
            })
            continue
        if not isinstance(matchups, list) or not matchups:
            raise ValueError(f"Sleeper week {week} matchup response is not a non-empty list.")
        if expected_team_count is not None and len(matchups) != expected_team_count:
            raise ValueError(
                f"Sleeper week {week} returned {len(matchups)} matchup teams; "
                f"expected {expected_team_count}."
            )
        current_scores, comparison_scores = _score_week(
            stats,
            players,
            scoring_settings,
            reception_points,
            te_premium,
            qb_reception_points,
        )
        directory_positions = players.set_index("sleeper_id")["position"].to_dict()
        directory_names = players.set_index("sleeper_id")["full_name"].to_dict()
        ids_seen: set[str] = set()
        rows_by_matchup: dict[str, list[dict]] = defaultdict(list)
        team_rows = []
        starter_count = 0
        missing_stat_count = 0
        unmapped_count = 0
        reconciliation_errors = []

        for matchup in matchups:
            if not isinstance(matchup, dict):
                raise ValueError(f"Sleeper week {week} contains a non-object matchup row.")
            roster_id = _normalized_id(matchup.get("roster_id"))
            matchup_id = _normalized_id(matchup.get("matchup_id"))
            if not roster_id or not matchup_id:
                raise ValueError(
                    f"Sleeper week {week} matchup row is missing roster_id or matchup_id."
                )
            if roster_id in ids_seen:
                raise ValueError(f"Sleeper week {week} has duplicate roster_id {roster_id}.")
            ids_seen.add(roster_id)
            starters = _valid_player_ids(matchup.get("starters"), "starters", week, roster_id)
            starter_points = matchup.get("starters_points")
            if not isinstance(starter_points, list) or len(starter_points) != len(starters):
                raise ValueError(
                    f"Sleeper week {week} roster {roster_id} has misaligned starters_points."
                )
            historical_roster = set(
                _valid_player_ids(matchup.get("players"), "players", week, roster_id)
            )
            clean_starters = [
                player_id for player_id in starters
                if player_id.upper() not in EMPTY_STARTER_IDS
            ]
            if len(clean_starters) != len(set(clean_starters)):
                raise ValueError(
                    f"Sleeper week {week} roster {roster_id} has duplicate starters."
                )
            outside_roster = set(clean_starters) - historical_roster
            if outside_roster:
                raise ValueError(
                    f"Sleeper week {week} roster {roster_id} has starters absent from "
                    "its historical roster snapshot: " + ", ".join(sorted(outside_roster))
                )
            unmapped = [player_id for player_id in clean_starters if player_id not in directory_positions]
            if unmapped:
                unmapped_count += len(unmapped)
                raise ValueError(
                    f"Sleeper week {week} has starter IDs missing from its player directory: "
                    + ", ".join(unmapped)
                )
            starter_count += len(clean_starters)
            missing_stat_count += sum(player_id not in stats for player_id in clean_starters)
            current_total = sum(current_scores.get(player_id, 0.0) for player_id in clean_starters)
            comparison_total = sum(
                comparison_scores.get(player_id, 0.0) for player_id in clean_starters
            )
            official_total = sum(
                _number(value, f"starters_points for roster {roster_id}") if value is not None else 0.0
                for player_id, value in zip(starters, starter_points)
                if player_id.upper() not in EMPTY_STARTER_IDS
            )
            official_team_total = _number(
                matchup.get("points", official_total), "points"
            )
            if abs(official_total - official_team_total) > 0.02:
                raise ValueError(
                    f"Sleeper week {week} roster {roster_id} starters_points do not "
                    f"reconcile to matchup points (difference "
                    f"{abs(official_total - official_team_total):.3f})."
                )
            error = abs(current_total - official_total)
            reconciliation_errors.append(error)
            if error > 0.02:
                raise ValueError(
                    f"Sleeper week {week} roster {roster_id} current scoring does not "
                    f"reconcile to starters_points (difference {error:.3f})."
                )
            team_row = {
                "Week": week,
                "Matchup ID": matchup_id,
                "Roster ID": roster_id,
                "Team": roster_labels.get(roster_id, f"Roster {roster_id}"),
                "Current points": current_total,
                "Comparison points": comparison_total,
                "Actual Sleeper points": official_team_total,
                "Historical roster player count": len(historical_roster),
                "Actual starter count": len(clean_starters),
            }
            team_rows.append(team_row)
            rows_by_matchup[matchup_id].append(team_row)
            for player_id in clean_starters:
                record_present = player_id in stats
                player_week_rows.append({
                    "Week": week,
                    "Roster ID": roster_id,
                    "Team": team_row["Team"],
                    "Sleeper ID": player_id,
                    "Position": directory_positions[player_id],
                    "Current points": current_scores.get(player_id, 0.0),
                    "Comparison points": comparison_scores.get(player_id, 0.0),
                    "Stat record present": record_present,
                })
                if not record_present:
                    missing_starter_rows.append({
                        "Week": week,
                        "Roster ID": roster_id,
                        "Team": team_row["Team"],
                        "Sleeper ID": player_id,
                        "Player": directory_names.get(player_id, ""),
                        "Position": directory_positions[player_id],
                    })

        if expected_team_count is not None and len(ids_seen) != expected_team_count:
            raise ValueError(
                f"Sleeper week {week} has {len(ids_seen)} unique rosters; "
                f"expected {expected_team_count}."
            )
        for matchup_id, pair in rows_by_matchup.items():
            if len(pair) != 2:
                raise ValueError(
                    f"Sleeper week {week} matchup {matchup_id} contains "
                    f"{len(pair)} teams; exactly two are required."
                )
            first, second = pair
            current_margin = first["Current points"] - second["Current points"]
            comparison_margin = first["Comparison points"] - second["Comparison points"]
            current_winner = (
                first["Roster ID"] if current_margin > 0 else
                second["Roster ID"] if current_margin < 0 else "TIE"
            )
            comparison_winner = (
                first["Roster ID"] if comparison_margin > 0 else
                second["Roster ID"] if comparison_margin < 0 else "TIE"
            )
            matchup_rows.append({
                "Week": week,
                "Matchup ID": matchup_id,
                "Roster 1 ID": first["Roster ID"],
                "Team 1": first["Team"],
                "Roster 2 ID": second["Roster ID"],
                "Team 2": second["Team"],
                "Current team 1 points": first["Current points"],
                "Current team 2 points": second["Current points"],
                "Comparison team 1 points": first["Comparison points"],
                "Comparison team 2 points": second["Comparison points"],
                "Current margin (team 1 - team 2)": current_margin,
                "Comparison margin (team 1 - team 2)": comparison_margin,
                "Current absolute margin": abs(current_margin),
                "Comparison absolute margin": abs(comparison_margin),
                "Current winner roster ID": current_winner,
                "Comparison winner roster ID": comparison_winner,
                "Outcome changed": current_winner != comparison_winner,
            })
        coverage_rows.append({
            "Week": week,
            "Stats records": len(stats),
            "Matchup team records": len(matchups),
            "Starter records": starter_count,
            "Starters without stat records": missing_stat_count,
            "Starters missing directory mapping": unmapped_count,
            "Maximum current-score reconciliation error": max(reconciliation_errors, default=0.0),
            "Included": True,
            "Reason": "Historical starters, roster snapshots, and current scores validated.",
        })
        team_week_rows.extend(team_rows)

    coverage = pd.DataFrame(coverage_rows)
    matchup_frame = pd.DataFrame(
        matchup_rows,
        columns=[
            "Week", "Matchup ID", "Roster 1 ID", "Team 1", "Roster 2 ID", "Team 2",
            "Current team 1 points", "Current team 2 points",
            "Comparison team 1 points", "Comparison team 2 points",
            "Current margin (team 1 - team 2)",
            "Comparison margin (team 1 - team 2)",
            "Current absolute margin", "Comparison absolute margin",
            "Current winner roster ID", "Comparison winner roster ID",
            "Outcome changed",
        ],
    )
    team_weeks = pd.DataFrame(
        team_week_rows,
        columns=[
            "Week", "Matchup ID", "Roster ID", "Team", "Current points",
            "Comparison points", "Actual Sleeper points",
            "Historical roster player count", "Actual starter count",
        ],
    )
    player_weeks = pd.DataFrame(
        player_week_rows,
        columns=[
            "Week", "Roster ID", "Team", "Sleeper ID", "Position",
            "Current points", "Comparison points", "Stat record present",
        ],
    )
    missing_starters = pd.DataFrame(
        missing_starter_rows,
        columns=["Week", "Roster ID", "Team", "Sleeper ID", "Player", "Position"],
    )
    if matchup_frame.empty or team_weeks.empty:
        return {
            "weekly_data_coverage": coverage,
            "weekly_matchup_rescoring": matchup_frame,
            "weekly_team_scores": team_weeks,
            "weekly_missing_starter_stats": missing_starters,
            "weekly_team_season_outcomes": pd.DataFrame(columns=[
                "Scoring format", "Roster ID", "Team", "Weeks played", "Wins",
                "Losses", "Ties", "Record", "Head-to-head win percentage",
                "All-play wins", "All-play losses", "All-play ties",
                "All-play record", "All-play win percentage",
                "Head-to-head minus all-play win percentage", "Points for",
                "Points against", "Points-for rank", "Matchup outcomes changed",
                "Weekly score standard deviation",
                "Weekly score coefficient of variation",
            ]),
            "weekly_competitiveness_summary": pd.DataFrame(columns=[
                "Scoring format", "Team-week observations", "Average weekly team score",
                "Median weekly team score", "Population standard deviation of team scores",
                "Average absolute matchup margin", "Median absolute matchup margin",
                "Margin standard deviation (signed)", "Largest absolute matchup margin",
                "Tied matchups", "Matchups decided by 5 points or fewer",
                "Matchup rate within 5 points (%)",
                "Matchups decided by 10 points or fewer",
                "Matchup rate within 10 points (%)",
                "Matchups decided by 15 points or fewer",
                "Matchup rate within 15 points (%)",
                "Outcome flips versus current scoring",
            ]),
            "weekly_position_volatility": pd.DataFrame(columns=[
                "Scoring format", "Position", "Actual starter player-week observations",
                "Mean starter points", "Median starter points",
                "Population standard deviation", "10th percentile", "90th percentile",
                "Coefficient of variation (mean > 0 only)",
            ]),
            "weekly_by_week_summary": pd.DataFrame(columns=[
                "Scoring format", "Week", "Matchups",
                "Average absolute matchup margin", "Median absolute matchup margin",
                "Matchups within 5 points", "Rate within 5 points (%)",
                "Matchups within 10 points", "Rate within 10 points (%)",
                "Matchups within 15 points", "Rate within 15 points (%)",
            ]),
        }

    outcomes = []
    for scoring_format, points_column in WEEKLY_FORMAT_COLUMNS:
        flip_counts = defaultdict(int)
        changed = matchup_frame.loc[matchup_frame["Outcome changed"]]
        for _, row in changed.iterrows():
            flip_counts[row["Roster 1 ID"]] += 1
            flip_counts[row["Roster 2 ID"]] += 1
        for roster_id, team_group in team_weeks.groupby("Roster ID", sort=False):
            team_group = team_group.sort_values("Week")
            points_for = pd.to_numeric(team_group[points_column], errors="coerce")
            points_against = []
            wins = losses = ties = 0
            for _, row in team_group.iterrows():
                match = matchup_frame.loc[
                    matchup_frame["Week"].eq(row["Week"])
                    & matchup_frame["Matchup ID"].eq(row["Matchup ID"])
                ].iloc[0]
                own_is_first = row["Roster ID"] == match["Roster 1 ID"]
                opponent = match[
                    "Current team 2 points" if own_is_first else "Current team 1 points"
                ] if scoring_format == "Current scoring" else match[
                    "Comparison team 2 points" if own_is_first else "Comparison team 1 points"
                ]
                points_against.append(float(opponent))
                winner = match[
                    "Current winner roster ID" if scoring_format == "Current scoring"
                    else "Comparison winner roster ID"
                ]
                if winner == "TIE":
                    ties += 1
                elif winner == row["Roster ID"]:
                    wins += 1
                else:
                    losses += 1

            all_play_wins = all_play_losses = all_play_ties = 0
            for _, row in team_group.iterrows():
                peers = team_weeks.loc[
                    team_weeks["Week"].eq(row["Week"])
                    & team_weeks["Roster ID"].ne(roster_id),
                    ["Roster ID", points_column],
                ]
                for peer_points in peers[points_column]:
                    if row[points_column] > peer_points:
                        all_play_wins += 1
                    elif row[points_column] < peer_points:
                        all_play_losses += 1
                    else:
                        all_play_ties += 1
            head_to_head_win_pct = (
                (wins + 0.5 * ties) / len(team_group) if len(team_group) else float("nan")
            )
            all_play_comparisons = all_play_wins + all_play_losses + all_play_ties
            all_play_win_pct = (
                (all_play_wins + 0.5 * all_play_ties) / all_play_comparisons
                if all_play_comparisons
                else float("nan")
            )
            outcomes.append({
                "Scoring format": scoring_format,
                "Roster ID": roster_id,
                "Team": team_group["Team"].iloc[0],
                "Weeks played": len(team_group),
                "Wins": wins,
                "Losses": losses,
                "Ties": ties,
                "Record": f"{wins}-{losses}-{ties}",
                "Head-to-head win percentage": head_to_head_win_pct,
                "All-play wins": all_play_wins,
                "All-play losses": all_play_losses,
                "All-play ties": all_play_ties,
                "All-play record": f"{all_play_wins}-{all_play_losses}-{all_play_ties}",
                "All-play win percentage": all_play_win_pct,
                "Head-to-head minus all-play win percentage": (
                    head_to_head_win_pct - all_play_win_pct
                    if pd.notna(head_to_head_win_pct) and pd.notna(all_play_win_pct)
                    else float("nan")
                ),
                "Points for": float(points_for.sum()),
                "Points against": float(sum(points_against)),
                "Points-for rank": int(points_for.sum()),
                "Matchup outcomes changed": flip_counts[roster_id],
                "Weekly score standard deviation": points_for.std(ddof=0),
                "Weekly score coefficient of variation": (
                    points_for.std(ddof=0) / points_for.mean()
                    if points_for.mean() > 0
                    else float("nan")
                ),
            })
    outcome_frame = pd.DataFrame(outcomes)
    outcome_frame["Points-for rank"] = outcome_frame.groupby("Scoring format")[
        "Points for"
    ].rank(method="min", ascending=False).astype(int)

    summary_rows = []
    by_week_rows = []
    changed_count = int(matchup_frame["Outcome changed"].sum())
    for scoring_format, points_column in WEEKLY_FORMAT_COLUMNS:
        margin_column = (
            "Current absolute margin"
            if scoring_format == "Current scoring"
            else "Comparison absolute margin"
        )
        signed_column = (
            "Current margin (team 1 - team 2)"
            if scoring_format == "Current scoring"
            else "Comparison margin (team 1 - team 2)"
        )
        margins = pd.to_numeric(matchup_frame[margin_column], errors="coerce")
        scores = pd.to_numeric(team_weeks[points_column], errors="coerce")
        row = {
            "Scoring format": scoring_format,
            "Team-week observations": len(scores),
            "Average weekly team score": scores.mean(),
            "Median weekly team score": scores.median(),
            "Population standard deviation of team scores": scores.std(ddof=0),
            "Average absolute matchup margin": margins.mean(),
            "Median absolute matchup margin": margins.median(),
            "Margin standard deviation (signed)": pd.to_numeric(
                matchup_frame[signed_column], errors="coerce"
            ).std(ddof=0),
            "Largest absolute matchup margin": margins.max(),
            "Tied matchups": int(margins.eq(0).sum()),
            "Matchups decided by 5 points or fewer": int(margins.le(5).sum()),
            "Matchup rate within 5 points (%)": float(margins.le(5).mean() * 100),
            "Matchups decided by 10 points or fewer": int(margins.le(10).sum()),
            "Matchup rate within 10 points (%)": float(margins.le(10).mean() * 100),
            "Matchups decided by 15 points or fewer": int(margins.le(15).sum()),
            "Matchup rate within 15 points (%)": float(margins.le(15).mean() * 100),
            "Outcome flips versus current scoring": changed_count,
        }
        summary_rows.append(row)
        for week, week_matchups in matchup_frame.groupby("Week", sort=True):
            week_margins = pd.to_numeric(week_matchups[margin_column], errors="coerce")
            by_week_rows.append({
                "Scoring format": scoring_format,
                "Week": int(week),
                "Matchups": len(week_matchups),
                "Average absolute matchup margin": week_margins.mean(),
                "Median absolute matchup margin": week_margins.median(),
                "Matchups within 5 points": int(week_margins.le(5).sum()),
                "Rate within 5 points (%)": float(week_margins.le(5).mean() * 100),
                "Matchups within 10 points": int(week_margins.le(10).sum()),
                "Rate within 10 points (%)": float(week_margins.le(10).mean() * 100),
                "Matchups within 15 points": int(week_margins.le(15).sum()),
                "Rate within 15 points (%)": float(week_margins.le(15).mean() * 100),
            })

    volatility_rows = []
    for scoring_format, points_column in WEEKLY_FORMAT_COLUMNS:
        if player_weeks.empty:
            continue
        for position, group in player_weeks.groupby("Position", sort=False):
            points = pd.to_numeric(group[points_column], errors="coerce")
            mean = points.mean()
            volatility_rows.append({
                "Scoring format": scoring_format,
                "Position": position,
                "Actual starter player-week observations": len(points),
                "Mean starter points": mean,
                "Median starter points": points.median(),
                "Population standard deviation": points.std(ddof=0),
                "10th percentile": points.quantile(0.1),
                "90th percentile": points.quantile(0.9),
                "Coefficient of variation (mean > 0 only)": (
                    points.std(ddof=0) / mean if mean > 0 else float("nan")
                ),
            })

    return {
        "weekly_data_coverage": coverage,
        "weekly_matchup_rescoring": matchup_frame,
        "weekly_team_scores": team_weeks,
        "weekly_missing_starter_stats": missing_starters,
        "weekly_team_season_outcomes": outcome_frame,
        "weekly_competitiveness_summary": pd.DataFrame(summary_rows),
        "weekly_position_volatility": pd.DataFrame(volatility_rows),
        "weekly_by_week_summary": pd.DataFrame(by_week_rows),
    }


def export_analysis_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: str,
) -> None:
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
