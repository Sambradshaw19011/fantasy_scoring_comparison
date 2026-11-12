import pandas as pd

ELIGIBLE_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF/DST"}


def eligible_players(
    players: pd.DataFrame,
    stats_ids: set[str],
) -> pd.DataFrame:
    """Keep active eligible entries and anyone with a current-season stats ID."""
    unexpected = set(players["position"].dropna()) - ELIGIBLE_POSITIONS
    if unexpected:
        raise ValueError(
            "The player universe contains unsupported positions: "
            + ", ".join(sorted(unexpected))
        )
    has_stats = players["sleeper_id"].isin(stats_ids)
    active = players["active"].fillna(False) | players["status"].eq("Active")
    return players[active | has_stats].copy()


def build_match_report(
    players: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    directory_ids: set[str],
) -> tuple[pd.DataFrame, int]:
    """Report direct Sleeper-ID matches and stats records absent from the directory."""
    stats_ids = set(stats)
    rows = []

    for _, player in players.iterrows():
        player_id = player["sleeper_id"]
        record = stats.get(player_id)
        if record is None:
            status = "no_stats_record"
        elif record:
            status = "matched_with_stats"
        else:
            status = "matched_no_stats"
        rows.append({
            "record_type": "eligible_player",
            "player_id": player_id,
            "full_name": player.get("full_name", ""),
            "position": player.get("position", ""),
            "status": status,
        })

    unmatched_ids = sorted(stats_ids - directory_ids)
    for player_id in unmatched_ids:
        rows.append({
            "record_type": "unmatched_stats_id",
            "player_id": player_id,
            "full_name": "",
            "position": "",
            "status": "not found in Sleeper player directory",
        })

    report = pd.DataFrame(
        rows,
        columns=["record_type", "player_id", "full_name", "position", "status"],
    )
    return report, len(stats_ids & directory_ids)
