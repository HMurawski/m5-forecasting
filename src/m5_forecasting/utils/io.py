from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__).resolve().parents[3]
)  # .../m5-forecasting/src/m5_forecasting/utils/io.py
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
DUCKDB_PATH = WAREHOUSE_DIR / "m5.duckdb"


def ensure_dirs() -> None:
    """Create expected folders if missing."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)


def get_env(name: str, default: str | None = None) -> str | None:
    """Safe env fetch with default."""
    return os.getenv(name, default)
