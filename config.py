import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

LEAGUE_ID = os.getenv("SLEEPER_LEAGUE_ID", "").strip()
SEASON_OVERRIDE = os.getenv("SEASON", "").strip()

POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF/DST"]
POSITIONAL_LIMIT = 100
OVERALL_LIMIT = 250
PLAYER_CACHE = PROJECT_ROOT / "data" / "raw" / "sleeper_players.json"
STATS_CACHE_TEMPLATE = PROJECT_ROOT / "data" / "raw" / "sleeper_stats_{season}.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Comparison format: 1.0 PPR for QB/TE, 0.5 PPR for RB/WR.
COMPARISON_QB_RECEPTION_POINTS = 1.0
COMPARISON_BASE_RECEPTION_POINTS = 0.5
COMPARISON_TE_PREMIUM = 0.5
COMPARISON_FORMAT_NAME = "QB/TE PPR + RB/WR half-PPR"
