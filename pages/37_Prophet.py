"""Chapter 37 -- Prophet: Multiple Seasonalities & Changepoints."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(37, "Prophet", part="VIII")

st.markdown(
    "ARIMA is like a Swiss watch: elegant, principled, and built on solid theory. "
    "Prophet is like a Toyota: it just *works*, and it works especially well on the "
    "kind of messy, seasonal, real-world time series that ARIMA requires extensive "
    "hand-tuning to handle. Originally built by Facebook for forecasting things like "
    "daily active users, Prophet handles **multiple seasonalities** (daily, weekly, "
    "yearly), **trend changepoints**, and **missing data** out of the box. It is not "
    "always the best model, but it is almost always the best *starting point*."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

concept_box(
    "How Prophet Works",
    "Prophet decomposes a time series as:<br>"
    "<b>y(t) = g(t) + s(t) + h(t) + e(t)</b><br>"
    "where g(t) is a piecewise-linear (or logistic) growth trend that can change "
    "slope at automatically detected 'changepoints,' s(t) is a Fourier-based seasonal "
    "component (it literally fits sine and cosine waves at yearly, weekly, and daily "
    "frequencies), h(t) captures holidays and special events, and e(t) is the error. "
    "The key insight is that this is a *regression* problem, not a traditional time "
    "series model -- which means it can handle irregular spacing and missing data "
    "gracefully."
)

formula_box(
    "Fourier Terms for Seasonality",
    r"s(t) = \sum_{n=1}^{N} \left( a_n \cos\!\left(\frac{2\pi n t}{P}\right) "
    r"+ b_n \sin\!\left(\frac{2\pi n t}{P}\right) \right)",
    "P is the period (365.25 for yearly, 7 for weekly). N controls the flexibility "
    "of the seasonal shape -- more terms means the seasonal pattern can have sharper "
    "peaks and valleys. For yearly temperature, N=10 is usually plenty."
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.subheader("Prophet Settings")
city = st.sidebar.selectbox("City", CITY_LIST, key="prophet_city")
forecast_days = st.sidebar.slider("Forecast Days", 14, 90, 30, key="prophet_horizon")
changepoint_scale = st.sidebar.slider(
    "Changepoint Prior Scale", 0.01, 0.5, 0.05, 0.01, key="prophet_cp"
)

# ── Prepare data in Prophet format ───────────────────────────────────────────
city_df = fdf[fdf["city"] == city].copy()
daily = (
    city_df.groupby("date")["temperature_c"]
    .mean()
    .reset_index()
    .sort_values("date")
)
daily["date"] = pd.to_datetime(daily["date"])
prophet_df = daily.rename(columns={"date": "ds", "temperature_c": "y"})

# Train/test split
n_test = min(forecast_days, len(prophet_df) // 5)
train_pr = prophet_df.iloc[:-n_test].copy()
test_pr = prophet_df.iloc[-n_test:].copy()

# ── Section 1: Fit Prophet ───────────────────────────────────────────────────
st.header(f"1. Fitting Prophet on {city}")

try:
    from prophet import Prophet

    with st.spinner("Fitting Prophet model..."):
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=changepoint_scale,
        )
        m.fit(train_pr)

    future = m.make_future_dataframe(periods=n_test)
    forecast = m.predict(future)

    prophet_available = True

except ImportError:
    st.error(
        "The `prophet` package is not installed. "
        "Install it with `pip install prophet` to enable this chapter."
    )
    prophet_available = False

if prophet_available:
    # ── Section 2: Forecast Plot ─────────────────────────────────────────────
    st.header("2. Forecast with Uncertainty Intervals")

    fig_fc = go.Figure()

    # Training data
    plot_start = max(0, len(train_pr) - 120)
    fig_fc.add_trace(go.Scatter(
        x=train_pr["ds"].iloc[plot_start:],
        y=train_pr["y"].iloc[plot_start:],
        mode="lines", name="Training Data",
        line=dict(color="#264653"),
    ))

    # Actual test
    fig_fc.add_trace(go.Scatter(
        x=test_pr["ds"], y=test_pr["y"],
        mode="lines", name="Actual",
        line=dict(color="#2A9D8F", width=2),
    ))

    # Prophet forecast (only future portion)
    fc_future = forecast[forecast["ds"] > train_pr["ds"].max()]
    fig_fc.add_trace(go.Scatter(
        x=fc_future["ds"], y=fc_future["yhat"],
        mode="lines", name="Prophet Forecast",
        line=dict(color="#E63946", width=2, dash="dash"),
    ))

    # Uncertainty
    fig_fc.add_trace(go.Scatter(
        x=list(fc_future["ds"]) + list(fc_future["ds"][::-1]),
        y=list(fc_future["yhat_upper"]) + list(fc_future["yhat_lower"][::-1]),
        fill="toself", fillcolor="rgba(230,57,70,0.15)",
        line=dict(width=0), name="Uncertainty (80%)",
    ))

    apply_common_layout(fig_fc, f"Prophet Forecast: {city}", 500)
    st.plotly_chart(fig_fc, use_container_width=True)

    # Metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    merged = test_pr.merge(fc_future[["ds", "yhat"]], on="ds", how="inner")
    if len(merged) > 0:
        mae = mean_absolute_error(merged["y"], merged["yhat"])
        rmse = np.sqrt(mean_squared_error(merged["y"], merged["yhat"]))
        c1, c2 = st.columns(2)
        c1.metric("Prophet MAE", f"{mae:.2f} deg C")
        c2.metric("Prophet RMSE", f"{rmse:.2f} deg C")

    # ── Section 3: Components Plot ───────────────────────────────────────────
    st.header("3. Components Plot")

    st.markdown(
        "This is where Prophet really shines. It automatically decomposes "
        "its forecast into interpretable components -- here, the trend (is the "
        "city getting warmer or cooler over our data window?) and the yearly "
        "seasonality (the familiar sinusoidal dance of the seasons). This "
        "decomposition is not just diagnostic; it is the actual model structure."
    )

    fig_comp = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Trend", "Yearly Seasonality"],
        shared_xaxes=False, vertical_spacing=0.15,
    )

    # Trend
    fig_comp.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["trend"],
        mode="lines", line=dict(color="#E63946"), showlegend=False,
    ), row=1, col=1)

    # Yearly seasonality
    yearly = forecast[["ds", "yearly"]].copy()
    yearly["day_of_year"] = yearly["ds"].dt.dayofyear
    yearly_avg = yearly.groupby("day_of_year")["yearly"].mean().reset_index()

    fig_comp.add_trace(go.Scatter(
        x=yearly_avg["day_of_year"], y=yearly_avg["yearly"],
        mode="lines", line=dict(color="#2A9D8F"), showlegend=False,
    ), row=2, col=1)

    fig_comp.update_xaxes(title_text="Date", row=1, col=1)
    fig_comp.update_xaxes(title_text="Day of Year", row=2, col=1)
    fig_comp.update_yaxes(title_text="Trend (deg C)", row=1, col=1)
    fig_comp.update_yaxes(title_text="Seasonal Effect (deg C)", row=2, col=1)
    fig_comp.update_layout(height=550, template="plotly_white")
    st.plotly_chart(fig_comp, use_container_width=True)

    insight_box(
        "The yearly seasonality component shows the familiar sinusoidal pattern: "
        "positive in summer (around days 150-250) and negative in winter. This "
        "is Prophet's Fourier terms doing their job -- fitting sine and cosine waves "
        "to the annual cycle. The trend captures any gradual drift over the data "
        "period, which for weather is usually small but nonzero."
    )

    # ── Section 4: Changepoints ──────────────────────────────────────────────
    st.header("4. Trend Changepoints")

    st.markdown(
        "One of Prophet's most clever features: it automatically detects points "
        "where the trend changes slope. Maybe the first two years showed gradual "
        "warming, then it plateaued. The `changepoint_prior_scale` parameter "
        "controls how willing Prophet is to add changepoints -- think of it as "
        "a skepticism dial. Low values mean 'I do not believe the trend really changed,' "
        "high values mean 'sure, the trend changed every few months.'"
    )

    fig_cp = go.Figure()
    fig_cp.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["trend"],
        mode="lines", name="Trend", line=dict(color="#264653"),
    ))
    for cp in m.changepoints:
        fig_cp.add_vline(x=cp, line_dash="dash", line_color="rgba(230,57,70,0.5)")

    apply_common_layout(fig_cp, "Trend with Changepoints", 400)
    st.plotly_chart(fig_cp, use_container_width=True)

    st.markdown(
        f"Prophet detected **{len(m.changepoints)}** changepoints "
        f"(changepoint_prior_scale = {changepoint_scale}). "
        "Try cranking the slider up or down to see how the trend line responds. "
        "A low value gives you a straighter, more conservative trend. A high value "
        "lets the trend wiggle around, which might capture real structure or might "
        "just be overfitting. As with most things in statistics, the truth is "
        "somewhere in the middle."
    )

    # ── Section 5: Prophet vs ARIMA ──────────────────────────────────────────
    st.header("5. Prophet vs ARIMA Comparison")

    st.markdown(
        "Let's put our money where our mouth is. We compare Prophet's forecast "
        "against a simple ARIMA(2,1,2) model on the same test set. This is not "
        "entirely fair to ARIMA -- a properly tuned SARIMA model would do better -- "
        "but it illustrates why Prophet is popular for quick-and-dirty forecasting."
    )

    try:
        from statsmodels.tsa.arima.model import ARIMA as ARIMA_Model

        train_ts = train_pr.set_index("ds")["y"].asfreq("D").interpolate()
        arima = ARIMA_Model(train_ts, order=(2, 1, 2))
        arima_res = arima.fit()
        arima_fc = arima_res.get_forecast(steps=n_test)
        arima_mean = arima_fc.predicted_mean
        arima_ci = arima_fc.conf_int(alpha=0.2)

        fig_vs = go.Figure()
        fig_vs.add_trace(go.Scatter(
            x=train_pr["ds"].iloc[plot_start:],
            y=train_pr["y"].iloc[plot_start:],
            mode="lines", name="Training", line=dict(color="#264653"),
        ))
        fig_vs.add_trace(go.Scatter(
            x=test_pr["ds"], y=test_pr["y"],
            mode="lines", name="Actual", line=dict(color="black", width=2),
        ))
        fig_vs.add_trace(go.Scatter(
            x=fc_future["ds"], y=fc_future["yhat"],
            mode="lines", name="Prophet",
            line=dict(color="#E63946", width=2, dash="dash"),
        ))
        fig_vs.add_trace(go.Scatter(
            x=arima_mean.index, y=arima_mean.values,
            mode="lines", name="ARIMA(2,1,2)",
            line=dict(color="#7209B7", width=2, dash="dot"),
        ))
        apply_common_layout(fig_vs, "Prophet vs ARIMA", 500)
        st.plotly_chart(fig_vs, use_container_width=True)

        # Side-by-side metrics
        merged_arima = test_pr.copy()
        merged_arima["ds_dt"] = pd.to_datetime(merged_arima["ds"])
        arima_vals = arima_mean.values[:len(test_pr)]
        if len(arima_vals) == len(test_pr):
            mae_arima = mean_absolute_error(test_pr["y"], arima_vals)
            rmse_arima = np.sqrt(mean_squared_error(test_pr["y"], arima_vals))
        else:
            mae_arima = rmse_arima = float("nan")

        comp_df = pd.DataFrame({
            "Model": ["Prophet", "ARIMA(2,1,2)"],
            "MAE (deg C)": [
                round(mae, 2) if len(merged) > 0 else "N/A",
                round(mae_arima, 2),
            ],
            "RMSE (deg C)": [
                round(rmse, 2) if len(merged) > 0 else "N/A",
                round(rmse_arima, 2),
            ],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        insight_box(
            "Prophet often outperforms simple ARIMA on seasonal data because it "
            "explicitly models the yearly cycle with Fourier terms -- it *knows* that "
            "summer is warm and winter is cold. Plain ARIMA has to discover the seasonal "
            "pattern indirectly through differencing and high-order lags, which requires "
            "either a lot of parameters or the seasonal ARIMA (SARIMA) extension."
        )

    except Exception as e:
        st.info(f"ARIMA comparison skipped: {e}")

    warning_box(
        "Prophet works best with daily or sub-daily data that has strong seasonal "
        "patterns -- which sounds like a narrow use case until you realize that "
        "describes a huge fraction of real-world time series. It may underperform "
        "on very short series (less than a year), purely random-walk data, or data "
        "without clear periodicity. Know your tool's happy path."
    )

    code_example("""
from prophet import Prophet

# Prepare data: columns must be 'ds' and 'y'
df_prophet = daily.rename(columns={'date': 'ds', 'temperature_c': 'y'})

m = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.05)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

# Access components
trend = forecast['trend']
yearly = forecast['yearly']
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "What mathematical approach does Prophet use to model seasonality?",
    [
        "ARIMA seasonal differencing",
        "Fourier series (sum of sines and cosines)",
        "Exponential smoothing",
        "Wavelet transform",
    ],
    1,
    "Prophet represents seasonal patterns as a sum of Fourier terms (sines and "
    "cosines at different frequencies). This is both mathematically elegant and "
    "practically flexible -- you can model any periodic shape by adding enough "
    "harmonics.",
    key="prophet_quiz",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "Prophet decomposes time series into trend + seasonality + holidays + error. This is a regression framework, not a traditional time series model, which gives it surprising flexibility.",
    "It uses Fourier terms for flexible seasonal modeling (yearly, weekly, daily). More Fourier terms = more flexible seasonal shape.",
    "Changepoints allow the trend to shift direction at data-driven breakpoints. The changepoint_prior_scale controls how eagerly Prophet adds them.",
    "The changepoint_prior_scale is arguably the most important tuning parameter. Too low = rigid trend that misses real shifts. Too high = overfitting.",
    "Prophet often beats simple ARIMA on data with strong, complex seasonality, but a properly tuned SARIMA can fight back. Use the right tool for the job.",
])
