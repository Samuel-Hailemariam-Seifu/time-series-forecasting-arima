# Time Series Forecasting with ARIMA

This project provides a practical ARIMA forecasting pipeline with:

- robust CSV loading and preprocessing
- automatic ARIMA `(p, d, q)` selection by AIC
- holdout test evaluation (MAE, RMSE, MAPE)
- future forecasting and CSV export
- Streamlit UI for interactive forecasting

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Prepare data

Create a CSV with at least two columns:

- `date` (parseable date/time)
- `value` (numeric target)

Example:

```csv
date,value
2022-01-01,120
2022-02-01,131
2022-03-01,128
```

## 3) Run forecasting

### Option A: Web UI (recommended)

```bash
streamlit run app.py
```

Then upload your CSV, adjust settings, run forecast, and download results.

### Option B: CLI

```bash
python arima_forecast.py --input your_data.csv --date-col date --value-col value --test-size 12 --forecast-steps 12
```

Optional arguments:

- `--freq M` set frequency manually (`D`, `W`, `M`, etc.)
- `--max-p`, `--max-d`, `--max-q` control ARIMA order search space
- `--output forecast_output.csv` set output file path

## 4) Output

The script prints selected ARIMA order and test metrics, then writes:

- test-set actual vs predicted rows
- future forecast rows

to `forecast_output.csv`.
