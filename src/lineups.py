from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

import pandas as pd

from src.scoring import STAT_KEY_MAP

PLAYER_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF/DST"}
POSITION_ALIASES = {"DEF": "DEF/DST", "DST": "DEF/DST", "D/ST": "DEF/DST"}
FLEX_ELIGIBILITY = {
    "FLEX": {"RB", "WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
    "WRRB_FLEX": {"RB", "WR"},
    "WR_TE_FLEX": {"WR", "TE"},
    "WRTE_FLEX": {"WR", "TE"},
    "REC_FLEX": {"RB", "WR", "TE"},
    "OP": {"QB", "RB", "WR", "TE"},
}
NONSTARTER_SLOTS = {"BN", "IR", "TAXI", "RESERVE"}
SCENARIOS = (
    ("A", "Current baseline", "current_points_per_game", "current"),
    ("B", "Scoring only", "comparison_points_per_game", "current"),
    ("C", "Lineup only", "current_points_per_game", "proposed"),
    ("D", "Combined proposal", "comparison_points_per_game", "proposed"),
)
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF/DST")


def _normalized_position(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    token = value.strip().upper()
    return POSITION_ALIASES.get(token, token)


def _slot_eligibility(slot: Any) -> frozenset[str] | None:
    token = str(slot).strip().upper()
    if token in NONSTARTER_SLOTS:
        return None
    if token in FLEX_ELIGIBILITY:
        return frozenset(FLEX_ELIGIBILITY[token])
    position = _normalized_position(token)
    if position in PLAYER_POSITIONS:
        return frozenset({position})
    raise ValueError(f"Unsupported Sleeper roster slot {slot!r}.")


def _starter_slots(roster_positions: Sequence[str]) -> list[tuple[str, frozenset[str]]]:
    if not isinstance(roster_positions, (list, tuple)):
        raise ValueError("Sleeper roster_positions must be a list or tuple.")
    slots = []
    counts: Counter[str] = Counter()
    for raw_slot in roster_positions:
        eligibility = _slot_eligibility(raw_slot)
        if eligibility is None:
            continue
        token = str(raw_slot).strip().upper()
        counts[token] += 1
        slots.append((f"{token} {counts[token]}", eligibility))
    return slots


def propose_roster_positions(roster_positions: Sequence[str]) -> list[str]:
    """Replace exactly one WR slot with a standard FLEX, preserving all else."""
    if not isinstance(roster_positions, (list, tuple)):
        raise ValueError("Sleeper roster_positions must be a list or tuple.")
    proposed = list(roster_positions)
    wr_indices = [
        index
        for index, slot in enumerate(proposed)
        if str(slot).strip().upper() == "WR"
    ]
    if not wr_indices:
        raise ValueError("The proposed lineup requires at least one WR starter.")
    proposed[wr_indices[-1]] = "FLEX"
    validate_roster_transformation(roster_positions, proposed)
    return proposed


def validate_roster_transformation(
    current: Sequence[str],
    proposed: Sequence[str],
) -> None:
    """Require only one WR-to-standard-FLEX change and equal starter counts."""
    current_slots = Counter(str(slot).strip().upper() for slot in current)
    proposed_slots = Counter(str(slot).strip().upper() for slot in proposed)
    expected = current_slots.copy()
    if expected["WR"] < 1:
        raise ValueError("The current lineup has no WR starter to replace.")
    expected["WR"] -= 1
    expected["FLEX"] += 1
    if expected != proposed_slots:
        raise ValueError(
            "The proposed roster must remove exactly one WR and add exactly "
            "one standard FLEX while preserving all other roster slots."
        )
    if len(_starter_slots(current)) != len(_starter_slots(proposed)):
        raise ValueError("Current and proposed starter counts must be equal.")


def starter_slot_comparison(roster_positions: Sequence[str]) -> pd.DataFrame:
    proposed = propose_roster_positions(roster_positions)
    tokens = sorted(
        {
            str(slot).strip().upper()
            for slot in (*roster_positions, *proposed)
            if _slot_eligibility(slot) is not None
        }
    )
    current_counts = Counter(str(slot).strip().upper() for slot in roster_positions)
    proposed_counts = Counter(str(slot).strip().upper() for slot in proposed)
    return pd.DataFrame(
        [
            {
                "Slot": slot,
                "Current starters": current_counts[slot],
                "Proposed starters": proposed_counts[slot],
                "Change": proposed_counts[slot] - current_counts[slot],
            }
            for slot in tokens
        ]
    )


def optimize_lineup(
    players: pd.DataFrame,
    roster_positions: Sequence[str],
    *,
    score_column: str = "points_per_game",
) -> pd.DataFrame:
    """Find a legal, maximum-scoring lineup using each player at most once."""
    required = {"sleeper_id", "position", score_column}
    missing = required - set(players.columns)
    if missing:
        raise ValueError(f"Lineup players are missing columns: {sorted(missing)}")
    if players["sleeper_id"].astype(str).duplicated().any():
        raise ValueError("Lineup player IDs must be unique.")

    slots = _starter_slots(roster_positions)
    candidates = []
    for _, row in players.iterrows():
        player_id = str(row["sleeper_id"]).strip()
        position = _normalized_position(row["position"])
        if position not in PLAYER_POSITIONS:
            raise ValueError(
                f"Player {player_id!r} has unsupported position {row['position']!r}."
            )
        value = pd.to_numeric(pd.Series([row[score_column]]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        value = float(value)
        if not isfinite(value):
            raise ValueError(f"Lineup score for player {player_id!r} must be finite.")
        candidates.append((player_id, position, value, row))
    candidates.sort(key=lambda player: player[0])

    # Each state represents filled slots. Processing one player at a time prevents
    # the same player from being assigned to more than one slot.
    states: dict[int, tuple[float, tuple[tuple[int, int], ...]]] = {0: (0.0, ())}

    def assignment_key(assignments: tuple[tuple[int, int], ...]) -> tuple:
        return tuple(
            (slot_index, candidates[player_index][0])
            for slot_index, player_index in sorted(assignments)
        )

    def prefer(
        candidate: tuple[float, tuple[tuple[int, int], ...]],
        existing: tuple[float, tuple[tuple[int, int], ...]] | None,
    ) -> bool:
        if existing is None or candidate[0] > existing[0] + 1e-9:
            return True
        if abs(candidate[0] - existing[0]) <= 1e-9:
            return assignment_key(candidate[1]) < assignment_key(existing[1])
        return False

    for player_index, (_, position, points, _) in enumerate(candidates):
        next_states = dict(states)
        for mask, (total, assignments) in states.items():
            for slot_index, (_, eligible_positions) in enumerate(slots):
                bit = 1 << slot_index
                if mask & bit or position not in eligible_positions:
                    continue
                new_mask = mask | bit
                candidate = (total + points, assignments + ((slot_index, player_index),))
                if prefer(candidate, next_states.get(new_mask)):
                    next_states[new_mask] = candidate
        states = next_states

    best_mask = 0
    best_state = states[0]
    for mask, state in states.items():
        filled = mask.bit_count()
        best_filled = best_mask.bit_count()
        if filled > best_filled or (
            filled == best_filled
            and prefer(state, best_state)
        ):
            best_mask, best_state = mask, state

    selected_by_slot = {
        slot_index: player_index
        for slot_index, player_index in best_state[1]
    }
    rows = []
    for slot_index, (slot_label, _) in enumerate(slots):
        player_index = selected_by_slot.get(slot_index)
        if player_index is None:
            rows.append({
                "slot_index": slot_index,
                "slot": slot_label,
                "slot_type": slot_label.rsplit(" ", 1)[0],
                "sleeper_id": None,
                "position": None,
                "points_per_game": None,
                "filled": False,
            })
            continue
        player_id, position, points, row = candidates[player_index]
        rows.append({
            "slot_index": slot_index,
            "slot": slot_label,
            "slot_type": slot_label.rsplit(" ", 1)[0],
            "sleeper_id": player_id,
            "position": position,
            "points_per_game": points,
            "filled": True,
            "full_name": row.get("full_name"),
            "roster_id": row.get("roster_id"),
            "Team": row.get("Team"),
            "season_points": row.get("season_points"),
        })
    return pd.DataFrame(rows)


def _weekly_game_counts(
    weekly_data: Sequence[Mapping[str, Any]] | None,
) -> dict[str, float]:
    counts: dict[str, float] = {}
    stat_fields = set(STAT_KEY_MAP)
    for week in weekly_data or ():
        stats = week.get("stats") or {}
        if not isinstance(stats, Mapping):
            raise ValueError("Weekly stats must be a player-ID mapping.")
        for raw_id, record in stats.items():
            if not isinstance(record, Mapping):
                continue
            player_id = str(raw_id).strip()
            active = pd.to_numeric(
                pd.Series([record.get("gms_active")]), errors="coerce"
            ).iloc[0]
            if pd.notna(active) and float(active) > 0:
                counts[player_id] = counts.get(player_id, 0.0) + float(active)
            elif any(key in stat_fields and value is not None for key, value in record.items()):
                counts[player_id] = counts.get(player_id, 0.0) + 1.0
    return counts


def _games_played(
    rostered_players: pd.DataFrame,
    stats_frame: pd.DataFrame,
    weekly_data: Sequence[Mapping[str, Any]] | None,
) -> tuple[pd.Series, pd.Series]:
    if stats_frame["sleeper_id"].astype(str).duplicated().any():
        raise ValueError("Season stats must have one row per Sleeper player ID.")
    if "gms_active" not in stats_frame:
        season_games = pd.Series(dtype=float)
    else:
        season_games = pd.to_numeric(
            stats_frame.set_index(stats_frame["sleeper_id"].astype(str))[
                "gms_active"
            ],
            errors="coerce",
        )
    weekly_games = _weekly_game_counts(weekly_data)
    ids = rostered_players["sleeper_id"].astype(str)
    games = []
    has_games_source = []
    stats_ids = set(stats_frame["sleeper_id"].astype(str))
    for player_id in ids:
        season_value = season_games.get(player_id, float("nan"))
        if pd.notna(season_value) and float(season_value) >= 0:
            games.append(float(season_value))
            has_games_source.append(True)
        elif player_id in weekly_games:
            games.append(weekly_games[player_id])
            has_games_source.append(True)
        else:
            games.append(0.0 if player_id not in stats_ids else float("nan"))
            has_games_source.append(False)
    return (
        pd.Series(games, index=rostered_players.index, dtype=float),
        pd.Series(has_games_source, index=rostered_players.index, dtype=bool),
    )


def _cutoff_metrics(
    lineup: pd.DataFrame,
    roster: pd.DataFrame,
    tolerance_values: Sequence[float],
) -> tuple[float, str | None, str | None, dict[float, int]]:
    filled = lineup.loc[lineup["filled"]].copy()
    bench = roster.loc[~roster["sleeper_id"].astype(str).isin(
        set(filled["sleeper_id"].dropna().astype(str))
    )]
    best_gap = float("nan")
    best_slot = None
    best_bench_id = None
    candidate_gaps: dict[str, float] = {}

    for _, starter in filled.iterrows():
        slot_type = str(starter["slot_type"])
        eligibility = _slot_eligibility(slot_type)
        eligible_bench = bench.loc[
            bench["position"].map(_normalized_position).isin(eligibility)
        ]
        if eligible_bench.empty:
            continue
        differences = (
            float(starter["points_per_game"])
            - pd.to_numeric(
                eligible_bench["points_per_game"], errors="coerce"
            )
        )
        differences = differences.loc[differences >= -1e-8]
        if differences.empty:
            continue
        closest_index = differences.idxmin()
        gap = max(0.0, float(differences.loc[closest_index]))
        if not isfinite(best_gap) or gap < best_gap:
            best_gap = gap
            best_slot = str(starter["slot"])
            best_bench_id = str(eligible_bench.loc[closest_index, "sleeper_id"])
        for player_id, difference in differences.items():
            normalized_id = str(eligible_bench.loc[player_id, "sleeper_id"])
            candidate_gaps[normalized_id] = min(
                candidate_gaps.get(normalized_id, float("inf")),
                max(0.0, float(difference)),
            )

    counts = {
        tolerance: sum(gap <= tolerance + 1e-9 for gap in candidate_gaps.values())
        for tolerance in tolerance_values
    }
    return best_gap, best_slot, best_bench_id, counts


def build_lineup_analysis(
    rostered_players: pd.DataFrame,
    stats_frame: pd.DataFrame,
    roster_positions: Sequence[str],
    *,
    weekly_data: Sequence[Mapping[str, Any]] | None = None,
    tolerances: Sequence[float] = (3.0, 5.0),
) -> dict[str, pd.DataFrame]:
    """Build roster-depth and four-scenario lineup reports from scored roster IDs."""
    required = {
        "sleeper_id", "roster_id", "Team", "full_name", "position",
        "current_points", "comparison_points",
    }
    missing = required - set(rostered_players.columns)
    if missing:
        raise ValueError(f"Rostered players are missing columns: {sorted(missing)}")
    if rostered_players["sleeper_id"].astype(str).duplicated().any():
        raise ValueError("Rostered Sleeper player IDs must be unique.")
    if not {"sleeper_id"}.issubset(stats_frame.columns):
        raise ValueError("Season stats must include Sleeper player IDs.")
    tolerance_values = tuple(float(value) for value in tolerances)
    if (
        not tolerance_values
        or any(not isfinite(value) or value < 0 for value in tolerance_values)
        or len(set(tolerance_values)) != len(tolerance_values)
    ):
        raise ValueError("Viable-option tolerances must be unique nonnegative values.")

    proposed_positions = propose_roster_positions(roster_positions)
    players = rostered_players.copy()
    players["sleeper_id"] = players["sleeper_id"].astype(str)
    players["roster_id"] = players["roster_id"].astype(str)
    players["position"] = players["position"].map(_normalized_position)
    if not players["position"].isin(PLAYER_POSITIONS).all():
        raise ValueError("Rostered players include unsupported lineup positions.")
    if stats_frame["sleeper_id"].astype(str).duplicated().any():
        raise ValueError("Season stats must have one row per Sleeper player ID.")
    stats_ids = set(stats_frame["sleeper_id"].astype(str))
    players["has_season_stats_record"] = players["sleeper_id"].isin(stats_ids)
    games, games_source_available = _games_played(
        players, stats_frame, weekly_data
    )
    players["games_played"] = games
    players["games_played_source_available"] = games_source_available

    for scoring, source_column in (
        ("current", "current_points"),
        ("comparison", "comparison_points"),
    ):
        points = pd.to_numeric(players[source_column], errors="coerce")
        if points.isna().any() or not points.map(isfinite).all():
            raise ValueError(f"{source_column} must contain finite values.")
        players[f"{scoring}_season_points"] = points.astype(float)
        players[f"{scoring}_points_per_game"] = points.div(
            players["games_played"].where(players["games_played"].gt(0))
        )

    team_order_frame = (
        players[["roster_id", "Team"]]
        .drop_duplicates()
        .sort_values("roster_id", key=lambda column: pd.to_numeric(column, errors="coerce"))
    )
    if team_order_frame["roster_id"].duplicated().any():
        raise ValueError("Each roster ID must map to exactly one team label.")
    teams = team_order_frame.to_dict("records")

    depth_rows = []
    for team in teams:
        team_pool = players.loc[players["roster_id"].eq(team["roster_id"])]
        for format_label, points_column, ppg_column in (
            ("Current scoring", "current_season_points", "current_points_per_game"),
            (
                "QB/TE PPR + RB/WR half-PPR",
                "comparison_season_points",
                "comparison_points_per_game",
            ),
        ):
            for position in POSITIONS:
                position_pool = team_pool.loc[team_pool["position"].eq(position)]
                active = position_pool.loc[position_pool[ppg_column].notna()].sort_values(
                    ppg_column, ascending=False, kind="mergesort"
                )
                ppg_values = pd.to_numeric(active[ppg_column], errors="coerce")
                depth_rows.append({
                    "Roster ID": team["roster_id"],
                    "Team": team["Team"],
                    "Scoring format": format_label,
                    "Position": position,
                    "Rostered player count": len(position_pool),
                    "Players with games played": len(active),
                    "Players without a usable per-game sample": len(position_pool) - len(active),
                    "Season-to-date points": float(
                        pd.to_numeric(position_pool[points_column], errors="coerce").sum()
                    ),
                    "Mean player points per game": (
                        float(ppg_values.mean()) if not ppg_values.empty else float("nan")
                    ),
                    "Top 2 mean points per game": (
                        float(ppg_values.head(2).mean()) if not ppg_values.empty else float("nan")
                    ),
                    "Players ranked 3-4 mean points per game": (
                        float(ppg_values.iloc[2:4].mean())
                        if len(ppg_values) > 2
                        else float("nan")
                    ),
                })

    slots_by_lineup = {
        "current": list(roster_positions),
        "proposed": proposed_positions,
    }
    scenario_rows = []
    selection_rows = []
    viable_rows = []
    for team in teams:
        team_pool = players.loc[players["roster_id"].eq(team["roster_id"])].copy()
        for scenario_id, scenario_label, ppg_column, lineup_name in SCENARIOS:
            season_column = (
                "current_season_points"
                if scenario_id in ("A", "C")
                else "comparison_season_points"
            )
            team_pool["season_points"] = team_pool[season_column]
            lineup = optimize_lineup(
                team_pool, slots_by_lineup[lineup_name], score_column=ppg_column
            )
            selected_ids = set(lineup["sleeper_id"].dropna().astype(str))
            cutoff_pool = team_pool.loc[team_pool[ppg_column].notna()].copy()
            cutoff_pool["points_per_game"] = cutoff_pool[ppg_column]
            gap, gap_slot, bench_id, counts = _cutoff_metrics(
                lineup,
                cutoff_pool,
                tolerance_values,
            )
            selected = lineup.loc[lineup["filled"]]
            contributions = {
                f"{position} points per game": float(
                    selected.loc[selected["position"].eq(position), "points_per_game"].sum()
                )
                for position in POSITIONS
            }
            flex_used = int(
                selected["slot_type"].eq("FLEX").sum()
            )
            superflex_used = int(
                selected["slot_type"].eq("SUPER_FLEX").sum()
            )
            summary = {
                "Roster ID": team["roster_id"],
                "Team": team["Team"],
                "Scenario": scenario_id,
                "Scenario label": scenario_label,
                "Scoring format": (
                    "Current scoring" if scenario_id in ("A", "C")
                    else "QB/TE PPR + RB/WR half-PPR"
                ),
                "Lineup format": (
                    "Current" if lineup_name == "current" else "Proposed"
                ),
                "Best lineup points per game": float(selected["points_per_game"].sum()),
                "Selected season-to-date points": float(
                    pd.to_numeric(selected["season_points"], errors="coerce").sum()
                ),
                "Starter slots": len(lineup),
                "Filled starter slots": len(selected),
                "Unfilled starter slots": int((~lineup["filled"]).sum()),
                "Standard FLEX slots used": flex_used,
                "SUPER_FLEX slots used": superflex_used,
                "Lowest selected points per game": (
                    float(selected["points_per_game"].min())
                    if not selected.empty else float("nan")
                ),
                "Best starter-to-bench gap (points per game)": gap,
                "Closest replacement slot": gap_slot,
                "Closest bench player ID": bench_id,
            }
            summary.update(contributions)
            for tolerance, count in counts.items():
                summary[f"Additional options within {tolerance:g} PPG"] = count
                viable_rows.append({
                    "Roster ID": team["roster_id"],
                    "Team": team["Team"],
                    "Scenario": scenario_id,
                    "Scenario label": scenario_label,
                    "Tolerance (points per game)": tolerance,
                    "Additional viable options": count,
                })
            scenario_rows.append(summary)

            for _, row in lineup.iterrows():
                selection_rows.append({
                    "Roster ID": team["roster_id"],
                    "Team": team["Team"],
                    "Scenario": scenario_id,
                    "Scenario label": scenario_label,
                    "Slot": row["slot"],
                    "Slot type": row["slot_type"],
                    "Sleeper player ID": row["sleeper_id"],
                    "Player": row["full_name"],
                    "Position": row["position"],
                    "Points per game": row["points_per_game"],
                    "Selected season-to-date points": row["season_points"],
                    "Filled": bool(row["filled"]),
                })

    scenarios = pd.DataFrame(scenario_rows)
    impact_rows = []
    for team in teams:
        team_scenarios = scenarios.loc[
            scenarios["Roster ID"].eq(team["roster_id"])
        ].set_index("Scenario")
        current = team_scenarios.loc["A"]
        scoring_only = team_scenarios.loc["B"]
        lineup_only = team_scenarios.loc["C"]
        combined = team_scenarios.loc["D"]
        impact_rows.append({
            "Roster ID": team["roster_id"],
            "Team": team["Team"],
            "Scoring-only lineup change (B minus A, points per game)": (
                scoring_only["Best lineup points per game"]
                - current["Best lineup points per game"]
            ),
            "Lineup-only change (C minus A, points per game)": (
                lineup_only["Best lineup points per game"]
                - current["Best lineup points per game"]
            ),
            "Combined proposal change (D minus A, points per game)": (
                combined["Best lineup points per game"]
                - current["Best lineup points per game"]
            ),
            "Interaction effect (D - B - C + A, points per game)": (
                combined["Best lineup points per game"]
                - scoring_only["Best lineup points per game"]
                - lineup_only["Best lineup points per game"]
                + current["Best lineup points per game"]
            ),
            **{
                f"Additional options within {tolerance:g} PPG change (D minus A)": (
                    combined[f"Additional options within {tolerance:g} PPG"]
                    - current[f"Additional options within {tolerance:g} PPG"]
                )
                for tolerance in tolerance_values
            },
            "Standard FLEX slots used change (D minus A)": (
                combined["Standard FLEX slots used"] - current["Standard FLEX slots used"]
            ),
        })

    flex_rows = []
    for team in teams:
        team_pool = players.loc[
            players["roster_id"].eq(team["roster_id"])
            & players["position"].isin({"RB", "WR", "TE"})
        ]
        for format_label, ppg_column in (
            ("Current scoring", "current_points_per_game"),
            ("QB/TE PPR + RB/WR half-PPR", "comparison_points_per_game"),
        ):
            active = team_pool.loc[team_pool[ppg_column].notna()]
            ordered = pd.to_numeric(active[ppg_column], errors="coerce").sort_values(
                ascending=False
            )
            flex_rows.append({
                "Roster ID": team["roster_id"],
                "Team": team["Team"],
                "Scoring format": format_label,
                "RB candidates with games": int(
                    active["position"].eq("RB").sum()
                ),
                "WR candidates with games": int(
                    active["position"].eq("WR").sum()
                ),
                "TE candidates with games": int(
                    active["position"].eq("TE").sum()
                ),
                "FLEX-eligible candidates with games": len(active),
                "Top 3 FLEX-eligible mean points per game": (
                    float(ordered.head(3).mean()) if not ordered.empty else float("nan")
                ),
                "Top 5 FLEX-eligible mean points per game": (
                    float(ordered.head(5).mean()) if not ordered.empty else float("nan")
                ),
            })

    slot_table = starter_slot_comparison(roster_positions)
    return {
        "lineup_slot_comparison": slot_table,
        "lineup_player_scores": players,
        "team_roster_depth": pd.DataFrame(depth_rows),
        "team_lineup_scenarios": scenarios,
        "team_lineup_impact": pd.DataFrame(impact_rows),
        "team_lineup_selections": pd.DataFrame(selection_rows),
        "team_viable_options": pd.DataFrame(viable_rows),
        "team_flex_candidates": pd.DataFrame(flex_rows),
    }
