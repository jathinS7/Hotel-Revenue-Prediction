"""Train an ADR (average daily rate) regressor and report honest test metrics.

Method:
1. Chronological split: the most recent 20% of bookings form the test set.
2. Compare a naive baseline (predict the training mean) with a linear model and
   a random forest, all in one preprocessing pipeline with no leakage.
3. Report MAE, RMSE and R2 on the test set, and save the best model.
"""

import json

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.data import chronological_split, prepare
from src.data_loader import load_raw_data


def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]),
         make_column_selector(dtype_include="number")),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore"))]),
         make_column_selector(dtype_exclude="number")),
    ])


def candidates():
    return {
        "Baseline (predict mean)": Pipeline([("prep", make_preprocessor()),
                                             ("model", DummyRegressor(strategy="mean"))]),
        "Ridge regression": Pipeline([("prep", make_preprocessor()),
                                      ("model", Ridge(alpha=1.0))]),
        "Random forest": Pipeline([("prep", make_preprocessor()),
                                   ("model", RandomForestRegressor(
                                       n_estimators=200, max_depth=14, min_samples_leaf=5,
                                       n_jobs=-1, random_state=config.RANDOM_STATE))]),
    }


def score(y_true, y_pred) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main():
    X, y, order = prepare(load_raw_data())
    X_train, X_test, y_train, y_test = chronological_split(X, y, order)
    print(f"Train: {len(X_train)} rows, test: {len(X_test)} rows")

    results, fitted = {}, {}
    for name, model in candidates().items():
        model.fit(X_train, y_train)
        results[name] = score(y_test, model.predict(X_test))
        fitted[name] = model
        print(f"{name:26s} {results[name]}")

    best = min(
        (n for n in results if n != "Baseline (predict mean)"),
        key=lambda n: results[n]["mae"],
    )
    print(f"\nBest model by test MAE: {best}")

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(fitted[best], config.MODEL_PATH)
    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(json.dumps({"best": best, "results": results}, indent=2))
    print(f"Saved {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
