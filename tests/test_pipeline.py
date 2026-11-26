import numpy as np
import pandas as pd

from src.data import booking_date, chronological_split, prepare
from src.train import make_preprocessor, score


def _toy(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "arrival_date_year": 2017,
        "arrival_date_month": "July",
        "arrival_date_day_of_month": rng.integers(1, 28, n),
        "lead_time": rng.integers(0, 100, n),
        "hotel": rng.choice(["City Hotel", "Resort Hotel"], n),
        "adults": rng.integers(1, 4, n),
        "children": rng.choice([0.0, np.nan], n),
        "company": np.nan,
        "reservation_status": "Check-Out",
        "reservation_status_date": "2017-07-01",
        "is_canceled": rng.integers(0, 2, n),
        "adr": rng.uniform(50, 200, n),
    })


def test_booking_date_is_arrival_minus_lead_time():
    df = pd.DataFrame({"arrival_date_year": [2017], "arrival_date_month": ["July"],
                       "arrival_date_day_of_month": [10], "lead_time": [9]})
    assert str(booking_date(df).iloc[0].date()) == "2017-07-01"


def test_leaky_columns_and_target_are_excluded_from_features():
    X, y, _ = prepare(_toy())
    for col in ["reservation_status", "reservation_status_date", "company", "is_canceled", "adr"]:
        assert col not in X.columns
    assert len(X) == len(y)


def test_split_is_chronological():
    X, y, order = prepare(_toy())
    X_tr, X_te, y_tr, y_te = chronological_split(X, y, order)
    assert len(X_tr) + len(X_te) == len(X)
    assert len(X_te) > 0


def test_preprocessor_handles_missing_values():
    X, y, _ = prepare(_toy())
    out = make_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()


def test_score_is_perfect_for_exact_predictions():
    s = score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert s == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}
