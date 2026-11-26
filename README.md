# Hotel Booking ADR Regression

Predicts the **average daily rate (ADR)** of a hotel booking, the price per night, from what is known when the booking is made.

This is the regression companion to the [cancellation classifier](../hotel-cancellation-prediction). It uses the same dataset: 119,390 bookings from two hotels, 2015–2017.

## Method

1. **Clean:** drop exact duplicates and rows with no ADR.
2. **Avoid leakage:** drop columns set after the stay (`reservation_status`, `reservation_status_date`), columns that are almost all missing (`company`), and `is_canceled`.
3. **Split by time:** order bookings by booking date and test on the most recent 20%. A random split would leak future seasonal pricing into training.
4. **Baseline first:** predict the training mean. Any model has to beat this to be useful.
5. **Compare:** Ridge regression and a Random Forest, both inside the same preprocessing pipeline (median imputation and scaling for numbers, one-hot encoding for categories).
6. **Score:** MAE, RMSE and R² on the test set. The model with the lowest test MAE is saved.

## Results

Run `python main.py` to produce the results on your copy of the data. They are saved to `outputs/metrics.json`.

No ADR results are quoted here yet. The code has only been tested on synthetic data, where the target was random, so those numbers mean nothing. The table in this README should be filled in from the real run.

A low R² would be a real finding. ADR depends on room type, season and market, and the features in this dataset explain only part of it.

## Project structure

```
data/            hotel_bookings.csv goes here (not committed)
models/          saved regressor (created by training)
outputs/         metrics.json (created by training)
src/
  config.py      paths, target and excluded columns
  data_loader.py loads the CSV and drops duplicates
  data.py        booking date, feature preparation, chronological split
  train.py       preprocessing, candidate models, evaluation, saving
tests/           unit tests for the data steps and scoring
main.py          entry point
```

## Setup

```bash
pip install -r requirements.txt
```

Download `hotel_bookings.csv` (the Hotel Booking Demand dataset) and put it in `data/`.

## Run

```bash
python main.py
pytest
```

## Limitations

- Only the ADR target is modelled. The original notebook explored cancellation, which is covered in the classifier project.
- Predictions are made before the booking is taken, so the model has no information on how the guest will behave later.

## Author

Chillagattu Jathin · [LinkedIn](https://www.linkedin.com/in/jathins7) · [Portfolio](https://jathindsprojects.netlify.app/)
