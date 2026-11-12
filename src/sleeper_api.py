import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.sleeper.app/v1"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "fantasy-scoring-comparison/1.0"})
ELIGIBLE_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF/DST"}
DEFENSE_POSITION_ALIASES = {"DEF", "DST", "D/ST", "DEF/DST"}


def get_json(url: str) -> Any:
    try:
        response = SESSION.get(url, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Unable to retrieve {url}: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"The response from {url} was not valid JSON.") from exc


def get_league(league_id: str) -> dict:
    league = get_json(f"{BASE_URL}/league/{league_id}")
    if not isinstance(league, dict):
        raise ValueError("Sleeper returned an unexpected league response.")
    return league


def get_nfl_state() -> dict:
    state = get_json(f"{BASE_URL}/state/nfl")
    if not isinstance(state, dict) or "season" not in state:
        raise ValueError("Sleeper returned an unexpected NFL state response.")
    return state


def get_league_rosters(league_id: str) -> list[dict]:
    """Retrieve the current roster ownership snapshot for a Sleeper league."""
    rosters = get_json(f"{BASE_URL}/league/{league_id}/rosters")
    if not isinstance(rosters, list) or any(
        not isinstance(roster, dict) for roster in rosters
    ):
        raise ValueError("Sleeper returned an unexpected league rosters response.")
    return rosters


def get_league_users(league_id: str) -> list[dict]:
    """Retrieve league users for team display labels."""
    users = get_json(f"{BASE_URL}/league/{league_id}/users")
    if not isinstance(users, list) or any(
        not isinstance(user, dict) for user in users
    ):
        raise ValueError("Sleeper returned an unexpected league users response.")
    return users


def get_week_matchups(
    league_id: str,
    week: int,
    cache_path: str | Path | None = None,
    refresh: bool = False,
) -> list[dict]:
    """Load one historical Sleeper matchup/lineup snapshot, using a local cache."""
    path = Path(cache_path) if cache_path else None
    if path and path.exists() and not refresh:
        try:
            matchups = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not read the cached Sleeper week {week} matchups at {path}. "
                "Use --refresh-weekly to replace it."
            ) from exc
    else:
        matchups = get_json(
            f"{BASE_URL}/league/{league_id}/matchups/{int(week)}"
        )

    if not isinstance(matchups, list) or any(
        not isinstance(matchup, dict) for matchup in matchups
    ):
        raise ValueError(
            f"Sleeper returned an unexpected matchup response for week {week}."
        )
    if path and matchups and (refresh or not path.exists()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(matchups), encoding="utf-8")
    return matchups


def get_all_players(
    cache_path: str | Path = "data/raw/sleeper_players.json",
    refresh: bool = False,
) -> dict:
    """Load the full Sleeper player map from disk or refresh it from the API."""
    path = Path(cache_path)
    if path.exists() and not refresh:
        try:
            player_map = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not read the cached Sleeper player directory at {path}. "
                "Use --refresh-players to replace it."
            ) from exc
    else:
        print("Downloading the full Sleeper NFL player directory...")
        player_map = get_json(f"{BASE_URL}/players/nfl")
        if not isinstance(player_map, dict):
            raise ValueError("Sleeper returned an unexpected player directory.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(player_map), encoding="utf-8")

    if not isinstance(player_map, dict):
        raise ValueError(f"The cached player directory at {path} is not a JSON object.")
    return player_map


def normalize_players(player_map: dict) -> pd.DataFrame:
    """Normalize eligible Sleeper positions and validate stable directory IDs."""
    rows = []
    for sleeper_id, player in player_map.items():
        if not isinstance(player, dict):
            continue
        raw_position = player.get("position")
        position = (
            "DEF/DST"
            if isinstance(raw_position, str)
            and raw_position.strip().upper() in DEFENSE_POSITION_ALIASES
            else raw_position.strip().upper()
            if isinstance(raw_position, str)
            else None
        )
        if position not in ELIGIBLE_POSITIONS:
            continue

        normalized_id = str(sleeper_id).strip() if sleeper_id is not None else ""
        if not normalized_id:
            raise ValueError(
                f"Sleeper directory contains an eligible {position} without a player ID."
            )
        team = player.get("team")
        if position == "DEF/DST" and (
            not isinstance(team, str) or not team.strip()
        ):
            raise ValueError(
                f"Sleeper defense {normalized_id} is missing its team abbreviation."
            )
        rows.append({
            "sleeper_id": normalized_id,
            "full_name": player.get("full_name") or " ".join(
                part for part in (player.get("first_name"), player.get("last_name"))
                if part
            ),
            "first_name": player.get("first_name"),
            "last_name": player.get("last_name"),
            "position": position,
            "team": team.strip().upper() if isinstance(team, str) else team,
            "age": player.get("age"),
            "years_exp": player.get("years_exp"),
            "status": player.get("status"),
            "active": bool(player.get("active", False)),
        })

    players = pd.DataFrame(rows)
    if players.empty:
        raise RuntimeError("Sleeper returned no eligible QB/RB/WR/TE/K/DEF players.")
    duplicate_ids = players.loc[
        players["sleeper_id"].duplicated(keep=False), "sleeper_id"
    ].unique()
    if len(duplicate_ids):
        raise ValueError(
            "Sleeper directory contains duplicate IDs after normalization: "
            + ", ".join(sorted(duplicate_ids))
        )
    defenses = players.loc[players["position"].eq("DEF/DST")]
    duplicate_defenses = defenses.loc[
        defenses["team"].duplicated(keep=False), ["team", "sleeper_id"]
    ]
    if not duplicate_defenses.empty:
        teams = sorted(duplicate_defenses["team"].unique())
        raise ValueError(
            "Sleeper directory contains multiple DEF/DST entries for team(s): "
            + ", ".join(teams)
        )
    return players.reset_index(drop=True)


def player_position_audit(player_map: dict) -> pd.DataFrame:
    """Count raw Sleeper directory positions and show their inclusion mapping."""
    counts = Counter()
    for player in player_map.values():
        if isinstance(player, dict):
            position = player.get("position")
            label = position.strip().upper() if isinstance(position, str) else ""
            counts[label or "<missing>"] += 1

    rows = []
    for raw_position, count in sorted(counts.items()):
        normalized = (
            "DEF/DST" if raw_position in DEFENSE_POSITION_ALIASES
            else raw_position if raw_position in ELIGIBLE_POSITIONS
            else ""
        )
        rows.append({
            "Sleeper directory position": raw_position,
            "Normalized position": normalized,
            "Directory entries": count,
            "Included": bool(normalized),
        })
    return pd.DataFrame(rows)
