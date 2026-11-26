"""Paths and constants shared by the training script."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "hotel_bookings.csv"
MODEL_PATH = BASE_DIR / "models" / "adr_regressor.pkl"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"

TARGET = "adr"  # average daily rate, the price per night
RANDOM_STATE = 42

# Excluded because they are leaky (set after the stay) or unused.
EXCLUDED = [
    "reservation_status", "reservation_status_date", "assigned_room_type",
    "booking_changes", "company", "is_canceled",
]
