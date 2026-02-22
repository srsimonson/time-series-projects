# app.py
import streamlit as st
import pandas as pd
import numpy as np

from prophet import Prophet
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go

st.set_page_config(page_title="Prophet Forecast (Weekly)", layout="wide")
st.title("Prophet Forecast Dashboard (Weekly Aggregation)")

# Avoid triple-quoted strings to prevent copy/paste syntax issues
st.markdown(
    "Upload a CSV that includes:\n\n"
    "- a **date/time** column (will be converted to datetime)\n"
    "- a numeric **units_sold** column\n\n"
    "The app will resample to a chosen frequency, fit Prophet, compute RMSE on a holdout set, and plot forecasts."
)

# ---------- Sidebar controls ----------
st.sidebar.header("Controls")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

freq = st.sidebar.selectbox(
    "Resample frequency",
    options=["W", "D", "M"],
    index=0,
    help="W=weekly, D=daily, M=monthly"
)

test_size_pct = st.sidebar.slider(
    "Test size (%)",
    min_value=10,
    max_value=50,
    value=20,
    step=5
)

yearly_seasonality = st.sidebar.checkbox("Yearly seasonality", value=True)
weekly_seasonality = st.sidebar.checkbox("Weekly seasonality", value=False)
daily_seasonality = st.sidebar.checkbox("Daily seasonality", value=False)

run_btn = st.sidebar.button("Run Prophet")

# ---------- Helper ----------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ---------- Main ----------
if uploaded is None:
    st.info("Upload a CSV to get started.")
    st.stop()

df = pd.read_csv(uploaded)

st.subheader("Preview")
st.dataframe(df.head(10), use_container_width=True)

if "units_sold" not in df.columns:
    st.error("Your CSV must include a column named 'units_sold'.")
    st.stop()

date_col = st.selectbox("Select your datetime column", options=list(df.columns))

# Parse datetime & set index
df = df.copy()
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

# Ensure numeric units_sold
df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
df = df.dropna(subset=["units_sold"])

# Aggregate to chosen frequency
series = df["units_sold"].resample(freq).sum()

# Prophet format: ds, y
prophet_data = series.reset_index()
prophet_data.columns = ["ds", "y"]

# Train/test split (time-ordered)
split_idx = int(len(prophet_data) * (1 - test_size_pct / 100.0))
train_df = prophet_data.iloc[:split_idx].copy()
test_df = prophet_data.iloc[split_idx:].copy()

st.subheader("Data shape")
c1, c2, c3 = st.columns(3)
c1.metric("Total rows", len(prophet_data))
c2.metric("Train rows", len(train_df))
c3.metric("Test rows", len(test_df))

if len(test_df) < 2:
    st.error("Test set is too small. Increase data length or reduce test size.")
    st.stop()

if not run_btn:
    st.warning("Click **Run Prophet** in the sidebar to fit + forecast.")
    st.stop()

with st.spinner("Fitting Prophet and forecasting…"):
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
    )
    model.fit(train_df)

    # Make future dataframe for exactly the test horizon
    future = model.make_future_dataframe(periods=len(test_df), freq=freq)
    forecast = model.predict(future)

    # Evaluate on test horizon (last N predictions)
    yhat_test = forecast["yhat"].iloc[-len(test_df):].to_numpy()
    y_test = test_df["y"].to_numpy()

    score = rmse(y_test, yhat_test)

st.success(f"Prophet RMSE: {score:,.4f}")

# -------- Plot ----------
st.subheader("Forecast plot (Actual vs Predicted)")

actual_train = go.Scatter(
    x=train_df["ds"], y=train_df["y"], name="Actual (Train)", mode="lines"
)
actual_test = go.Scatter(
    x=test_df["ds"], y=test_df["y"], name="Actual (Test)", mode="lines"
)
predicted = go.Scatter(
    x=forecast["ds"], y=forecast["yhat"], name="Predicted (yhat)", mode="lines"
)
upper = go.Scatter(
    x=forecast["ds"], y=forecast["yhat_upper"], name="Upper", mode="lines"
)
lower = go.Scatter(
    x=forecast["ds"], y=forecast["yhat_lower"], name="Lower", mode="lines"
)

fig = go.Figure([actual_train, actual_test, predicted, upper, lower])
fig.update_layout(
    title="Prophet Forecast with Uncertainty Interval",
    xaxis_title="Date",
    yaxis_title="Units Sold",
    height=600,
)

st.plotly_chart(fig, use_container_width=True)

# -------- Table ----------
st.subheader("Forecast table (last 25 rows)")
st.dataframe(
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(25),
    use_container_width=True
)
