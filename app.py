from io import StringIO
import warnings

import pandas as pd
import streamlit as st

from arima_forecast import load_series, run_forecast_pipeline


st.set_page_config(page_title="ARIMA Forecaster", page_icon="📈", layout="wide")
st.title("ARIMA Time Series Forecaster")
st.caption("Upload a CSV, tune ARIMA settings, and generate forecasts.")

with st.sidebar:
    st.header("Settings")
    date_col = st.text_input("Date column", value="date")
    value_col = st.text_input("Value column", value="value")
    freq = st.text_input("Frequency (optional)", value="")
    test_size = st.number_input("Test size", min_value=1, max_value=1000, value=12, step=1)
    forecast_steps = st.number_input(
        "Forecast steps", min_value=1, max_value=1000, value=12, step=1
    )
    max_p = st.slider("Max p", min_value=0, max_value=8, value=4)
    max_d = st.slider("Max d", min_value=0, max_value=3, value=2)
    max_q = st.slider("Max q", min_value=0, max_value=8, value=4)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Data Preview")
st.dataframe(df.head(15), use_container_width=True)

if st.button("Run Forecast", type="primary"):
    try:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        with warnings.catch_warnings():
            # ARIMA grid search can emit many numeric convergence warnings.
            warnings.simplefilter("ignore")
            series = load_series(
                csv_path=csv_buffer,
                date_col=date_col,
                value_col=value_col,
                freq=freq.strip() or None,
            )
            best_order, metrics, out_df = run_forecast_pipeline(
                series=series,
                test_size=int(test_size),
                forecast_steps=int(forecast_steps),
                max_p=int(max_p),
                max_d=int(max_d),
                max_q=int(max_q),
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best order", str(best_order))
        c2.metric("MAE", f"{metrics['mae']:.4f}")
        c3.metric("RMSE", f"{metrics['rmse']:.4f}")
        c4.metric("MAPE", f"{metrics['mape']:.2f}%")

        st.subheader("Forecast Plot")
        history_df = pd.DataFrame({"date": series.index, "value": series.values})
        st.line_chart(history_df.set_index("date"))

        plot_df = out_df.copy()
        plot_df["actual"] = pd.to_numeric(plot_df["actual"], errors="coerce")
        plot_df["predicted"] = pd.to_numeric(plot_df["predicted"], errors="coerce")

        st.write("Test and Future Predictions")
        st.line_chart(
            plot_df.set_index("date")[["actual", "predicted"]].rename(
                columns={"actual": "Actual", "predicted": "Predicted"}
            )
        )

        st.subheader("Output Table")
        st.dataframe(out_df, use_container_width=True)

        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download forecast_output.csv",
            data=csv_bytes,
            file_name="forecast_output.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Forecasting failed: {exc}")
