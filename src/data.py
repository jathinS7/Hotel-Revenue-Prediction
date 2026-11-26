"""Feature preparation and a chronological train/test split."""

import pandas as pd

from src import config
from src.data_loader import load_raw_data

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def booking_date(df: pd.DataFrame) -> pd.Series:
    """Arrival date minus lead time. Used only to order the rows in time."""
    arrival = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" + df["arrival_date_month"]
        + "-" + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )
    return arrival - pd.to_timedelta(df["lead_time"], unit="D")


def prepare(df: pd.DataFrame):
    """Return (X, y, order) where order is the booking date for sorting.

    Drops rows without a price (ADR), since that is the target.
    """
    df = df.dropna(subset=[config.TARGET]).copy()
    df["_booking_date"] = booking_date(df)
    df = df.sort_values("_booking_date").reset_index(drop=True)
    y = df[config.TARGET]
    X = df.drop(columns=[config.TARGET, "_booking_date", *config.EXCLUDED], errors="ignore")
    return X, y, df["_booking_date"]


def chronological_split(X, y, order, test_frac=0.2):
    """Train on the earlier bookings and test on the most recent ones."""
    cut = int(len(X) * (1 - test_frac))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
