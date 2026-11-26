"""Loads the raw hotel bookings dataset."""

import pandas as pd

from src import config


def load_raw_data(path=config.DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and drop exact duplicate rows."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Place hotel_bookings.csv in data/.")
    return pd.read_csv(path).drop_duplicates()
