import pandas as pd

from src.lineups import (
    build_lineup_analysis,
    optimize_lineup,
    propose_roster_positions,
    starter_slot_comparison,
)
from src.scoring import score_with_sleeper_settings


CURRENT_SLOTS = [
    "QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "SUPER_FLEX",
    "K", "DEF", "BN", "BN",
]


def test_proposed_lineup_replaces_one_wr_and_preserves_superflex_and_bench():
    proposed = propose_roster_positions(CURRENT_SLOTS)
    assert proposed == [
        "QB", "RB", "RB", "WR", "WR", "FLEX", "TE", "FLEX",
        "SUPER_FLEX", "K", "DEF", "BN", "BN",
    ]

    table = starter_slot_comparison(CURRENT_SLOTS).set_index("Slot")
    assert table.loc["WR", "Change"] == -1
    assert table.loc["FLEX", "Change"] == 1
    assert table.loc["SUPER_FLEX", "Change"] == 0
    assert table["Current starters"].sum() == table["Proposed starters"].sum()


def test_optimizer_assigns_each_player_at_most_once_and_fills_legal_slots():
    players = pd.DataFrame([
        {"sleeper_id": "rb-1", "position": "RB", "points_per_game": 20.0},
        {"sleeper_id": "wr-1", "position": "WR", "points_per_game": 15.0},
    ])
    lineup = optimize_lineup(
        players, ["RB", "FLEX"], score_column="points_per_game"
    )
    selected = lineup.loc[lineup["filled"]]
    assert len(selected) == 2
    assert selected["sleeper_id"].nunique() == 2
    assert set(zip(selected["slot_type"], selected["position"])) == {
        ("RB", "RB"), ("FLEX", "WR")
    }


def test_standard_flex_and_superflex_have_distinct_eligibility():
    players = pd.DataFrame([
        {"sleeper_id": "qb-1", "position": "QB", "points_per_game": 35.0},
        {"sleeper_id": "wr-1", "position": "WR", "points_per_game": 20.0},
    ])
    lineup = optimize_lineup(
        players, ["FLEX", "SUPER_FLEX"], score_column="points_per_game"
    )
    selected = lineup.set_index("slot_type")
    assert selected.loc["FLEX", "position"] == "WR"
    assert selected.loc["SUPER_FLEX", "position"] == "QB"


def test_optimizer_reports_empty_positions_without_fabricating_players():
    empty = pd.DataFrame(columns=["sleeper_id", "position", "points_per_game"])
    lineup = optimize_lineup(
        empty, ["QB", "TE", "FLEX"], score_column="points_per_game"
    )
    assert len(lineup) == 3
    assert not lineup["filled"].any()
    assert lineup["sleeper_id"].isna().all()


def test_lineup_scenarios_use_identical_rosters_and_test_tolerance_sensitivity():
    rows = [
        ("q1", "QB", 25, 1), ("q2", "QB", 10, 1),
        ("r1", "RB", 20, 1), ("r2", "RB", 16, 1), ("r3", "RB", 13, 1),
        ("w1", "WR", 22, 1), ("w2", "WR", 17, 1), ("w3", "WR", 12, 1),
        ("w4", "WR", 11, 1), ("t1", "TE", 14, 1), ("t2", "TE", 10, 1),
        ("k1", "K", 6, 1), ("d1", "DEF/DST", 7, 1),
    ]
    rostered = pd.DataFrame([
        {
            "sleeper_id": player_id,
            "roster_id": "1",
            "Team": "Team One",
            "full_name": player_id,
            "position": position,
            "current_points": float(points),
            "comparison_points": float(points - 0.5 if position in {"RB", "WR"} else points),
        }
        for player_id, position, points, _ in rows
    ])
    stats = pd.DataFrame([
        {"sleeper_id": player_id, "gms_active": games}
        for player_id, _, _, games in rows
    ])
    analysis = build_lineup_analysis(
        rostered,
        stats,
        CURRENT_SLOTS,
        tolerances=(3, 5),
    )

    scenarios = analysis["team_lineup_scenarios"]
    assert scenarios["Scenario"].tolist() == ["A", "B", "C", "D"]
    assert scenarios["Starter slots"].nunique() == 1
    assert scenarios["Starter slots"].iloc[0] == 11
    assert scenarios["Unfilled starter slots"].eq(0).all()
    selections = analysis["team_lineup_selections"]
    for _, scenario in selections.groupby("Scenario"):
        selected_ids = scenario.loc[scenario["Filled"], "Sleeper player ID"]
        assert selected_ids.is_unique
    viable = analysis["team_viable_options"]
    wide = viable.loc[
        viable["Tolerance (points per game)"].eq(5),
        "Additional viable options",
    ].iloc[0]
    narrow = viable.loc[
        viable["Tolerance (points per game)"].eq(3),
        "Additional viable options",
    ].iloc[0]
    assert wide >= narrow


def test_comparison_scoring_changes_only_reception_points():
    players = pd.DataFrame({
        "position": ["QB", "RB", "WR", "TE", "K", "DEF/DST"],
        "rec": [1, 1, 1, 1, 0, 0],
        "pass_yd": [100, 0, 0, 0, 0, 0],
        "rush_yd": [0, 10, 0, 0, 0, 0],
        "rec_yd": [0, 0, 10, 10, 0, 0],
    })
    current = score_with_sleeper_settings(
        players,
        {"rec": 1.0, "pass_yd": 0.04, "rush_yd": 0.1, "rec_yd": 0.1},
    )
    comparison = score_with_sleeper_settings(
        players,
        {"rec": 1.0, "pass_yd": 0.04, "rush_yd": 0.1, "rec_yd": 0.1},
        reception_points=0.5,
        te_premium=1.0,
    )
    assert (comparison - current).tolist() == [-0.5, -0.5, -0.5, 0.5, 0.0, 0.0]


def test_missing_games_played_uses_weekly_evidence_and_keeps_zero_stat_players():
    rostered = pd.DataFrame([
        {
            "sleeper_id": "played",
            "roster_id": "1",
            "Team": "Team One",
            "full_name": "Played",
            "position": "RB",
            "current_points": 5.0,
            "comparison_points": 4.5,
        },
        {
            "sleeper_id": "no-stats",
            "roster_id": "1",
            "Team": "Team One",
            "full_name": "No Stats",
            "position": "RB",
            "current_points": 0.0,
            "comparison_points": 0.0,
        },
    ])
    stats = pd.DataFrame([{"sleeper_id": "played", "rush_yd": 50.0}])
    analysis = build_lineup_analysis(
        rostered,
        stats,
        ["WR", "RB"],
        weekly_data=[{"week": 1, "stats": {"played": {"rush_yd": 50.0}}}],
    )

    player_scores = analysis["lineup_player_scores"].set_index("sleeper_id")
    assert player_scores.loc["played", "games_played"] == 1
    assert player_scores.loc["played", "games_played_source_available"]
    assert pd.isna(player_scores.loc["no-stats", "current_points_per_game"])
    assert player_scores.loc["no-stats", "current_season_points"] == 0
    assert not player_scores.loc["no-stats", "has_season_stats_record"]
