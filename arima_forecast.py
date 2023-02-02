import argparse
from itertools import product
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ARIMA on a univariate time series and forecast future values."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV file containing date and value columns.",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default="date",
        help="Name of date column in CSV.",
    )
    parser.add_argument(
        "--value-col",
        type=str,
        default="value",
        help="Name of numeric target/value column in CSV.",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default=None,
        help="Optional Pandas frequency (e.g., D, W, M). If omitted, inferred.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=12,
        help="Number of final observations used as holdout test set.",
    )
    parser.add_argument(
        "--forecast-steps",
        type=int,
        default=12,
        help="Number of future time steps to forecast.",
    )
    parser.add_argument(
        "--max-p",
        type=int,
        default=4,
        help="Max AR order p for grid search.",
    )
    parser.add_argument(
        "--max-d",
        type=int,
        default=2,
        help="Max differencing order d for grid search.",
    )
    parser.add_argument(
        "--max-q",
        type=int,
        default=4,
        help="Max MA order q for grid search.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="forecast_output.csv",
        help="Output CSV path for test predictions and future forecast.",
    )
    return parser.parse_args()


def load_series(csv_path: Path, date_col: str, value_col: str, freq: str | None) -> pd.Series:
    df = pd.read_csv(csv_path)
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{date_col}' and '{value_col}'. "
            f"Found: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)
    df = df.drop_duplicates(subset=[date_col], keep="last")
    if df.empty:
        raise ValueError("No valid rows after parsing date/value columns.")

    series = df.set_index(date_col)[value_col].astype(float)

    if freq:
        series = series.asfreq(freq)
    else:
        inferred = pd.infer_freq(series.index)
        if inferred is not None:
            series = series.asfreq(inferred)

    # Fill occasional gaps with time interpolation to keep ARIMA stable.
    if series.isna().any():
        series = series.interpolate(method="time").ffill().bfill()

    return series


def train_test_split(series: pd.Series, test_size: int) -> Tuple[pd.Series, pd.Series]:
    if test_size <= 0 or test_size >= len(series):
        raise ValueError(
            f"test_size must be > 0 and < series length ({len(series)}). Got {test_size}."
        )
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    return train, test


def select_best_order(train: pd.Series, max_p: int, max_d: int, max_q: int) -> Tuple[int, int, int]:
    best_aic = np.inf
    best_order = None

    for p, d, q in product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
        if p == 0 and d == 0 and q == 0:
            continue
        try:
            model = ARIMA(train, order=(p, d, q))
            result = model.fit()
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = (p, d, q)
        except Exception:
            # Some combinations fail due to non-invertibility/non-stationarity.
            continue

    if best_order is None:
        raise RuntimeError("Could not fit any ARIMA order in the provided search space.")

    return best_order


def evaluate(test: pd.Series, predicted: pd.Series) -> dict:
    y_true = test.values
    y_pred = predicted.values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def run_forecast_pipeline(
    series: pd.Series,
    test_size: int,
    forecast_steps: int,
    max_p: int,
    max_d: int,
    max_q: int,
) -> tuple[tuple[int, int, int], dict, pd.DataFrame]:
    train, test = train_test_split(series, test_size)
    best_order = select_best_order(train, max_p, max_d, max_q)

    model = ARIMA(train, order=best_order)
    fitted = model.fit()

    test_pred = fitted.forecast(steps=len(test))
    test_pred.index = test.index
    metrics = evaluate(test, test_pred)

    # Retrain on full series for future forecasting.
    final_model = ARIMA(series, order=best_order).fit()
    future_forecast = final_model.forecast(steps=forecast_steps)

    out_df_test = pd.DataFrame(
        {
            "date": test.index,
            "actual": test.values,
            "predicted": test_pred.values,
            "segment": "test",
        }
    )
    out_df_future = pd.DataFrame(
        {
            "date": future_forecast.index,
            "actual": np.nan,
            "predicted": future_forecast.values,
            "segment": "future",
        }
    )
    out_df = pd.concat([out_df_test, out_df_future], ignore_index=True)
    return best_order, metrics, out_df


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    series = load_series(
        csv_path=input_path,
        date_col=args.date_col,
        value_col=args.value_col,
        freq=args.freq,
    )

    best_order, metrics, out_df = run_forecast_pipeline(
        series=series,
        test_size=args.test_size,
        forecast_steps=args.forecast_steps,
        max_p=args.max_p,
        max_d=args.max_d,
        max_q=args.max_q,
    )
    out_df.to_csv(args.output, index=False)

    print(f"Loaded observations: {len(series)}")
    print(f"Train size: {len(series) - args.test_size} | Test size: {args.test_size}")
    print(f"Selected ARIMA order: {best_order}")
    print(
        f"Test metrics -> MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, "
        f"MAPE: {metrics['mape']:.2f}%"
    )
    print(f"Saved forecast results to: {args.output}")


if __name__ == "__main__":
    main()
