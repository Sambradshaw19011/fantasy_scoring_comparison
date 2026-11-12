import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from src.sleeper_api import get_json

STATS_URL_TEMPLATE = "https://api.sleeper.app/v1/stats/nfl/regular/{season}"
WEEKLY_STATS_URL_TEMPLATE = (
    "https://api.sleeper.app/v1/stats/nfl/regular/{season}/{week}"
)


def validate_stats_response(
    payload: Any,
    *,
    allow_empty: bool = False,
) -> dict[str, dict[str, float]]:
    """Validate Sleeper's player-ID to raw-stat-object response."""
    if not isinstance(payload, dict) or (not payload and not allow_empty):
        raise ValueError(
            "Sleeper stats response must be a non-empty object keyed by player ID."
        )

    validated: dict[str, dict[str, float]] = {}
    for player_id, record in payload.items():
        if not isinstance(player_id, str) or not player_id.strip():
            raise ValueError("Sleeper stats response contains an invalid player ID.")
        if not isinstance(record, dict):
            raise ValueError(
                f"Sleeper stats record for player ID {player_id} is not an object."
            )

        clean_record: dict[str, float] = {}
        for stat_key, value in record.items():
            if not isinstance(stat_key, str) or not stat_key:
                raise ValueError(
                    f"Sleeper stats record for player ID {player_id} "
                    "contains an invalid stat key."
                )
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(
                    f"Sleeper stat {stat_key!r} for player ID {player_id} "
                    "must be numeric."
                )
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"Sleeper stat {stat_key!r} for player ID {player_id} "
                    "must be finite."
                )
            clean_record[stat_key] = numeric_value
        validated[player_id.strip()] = clean_record
    return validated


def fetch_season_stats(
    season: int,
    cache_path: str | Path | None = None,
    refresh: bool = False,
) -> dict[str, dict[str, float]]:
    """Load cached Sleeper regular-season stats or retrieve them from the API."""
    path = Path(cache_path) if cache_path else Path(
        f"data/raw/sleeper_stats_{int(season)}.json"
    )
    if path.exists() and not refresh:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not read the cached Sleeper stats at {path}. "
                "Use --refresh-stats to download a fresh response."
            ) from exc
    else:
        url = STATS_URL_TEMPLATE.format(season=int(season))
        print(f"Retrieving season statistics from Sleeper for {season}...")
        payload = get_json(url)

    stats = validate_stats_response(payload)
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats), encoding="utf-8")
    return stats


def fetch_weekly_stats(
    season: int,
    week: int,
    cache_path: str | Path | None = None,
    refresh: bool = False,
) -> dict[str, dict[str, float]]:
    """Load one Sleeper regular-season week's raw statistics or retrieve them."""
    path = Path(cache_path) if cache_path else Path(
        f"data/raw/sleeper_weekly_stats_{int(season)}_{int(week)}.json"
    )
    if path.exists() and not refresh:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not read the cached Sleeper week {week} stats at {path}. "
                "Use --refresh-weekly to replace it."
            ) from exc
    else:
        url = WEEKLY_STATS_URL_TEMPLATE.format(
            season=int(season),
            week=int(week),
        )
        print(f"Retrieving weekly statistics from Sleeper for {season}, week {week}...")
        payload = get_json(url)

    stats = validate_stats_response(payload, allow_empty=True)
    if stats and (refresh or not path.exists()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats), encoding="utf-8")
    elif not stats and path.exists() and refresh:
        path.unlink()
    return stats


def stats_to_frame(stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert Sleeper's player-keyed stats object to rows without changing IDs."""
    validated = validate_stats_response(stats)
    frame = pd.DataFrame.from_dict(validated, orient="index")
    frame.index.name = "sleeper_id"
    return frame.reset_index()
