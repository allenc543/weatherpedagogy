"""Chapter 36 -- ARIMA Modeling and Forecasting."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(36, "ARIMA", part="VIII")

st.markdown(
    "You are probably wondering why we would bother with ARIMA when Prophet exists "
    "and machine learning can do anything. Fair question. The answer is that ARIMA is "
    "the *foundation* -- the intellectual bedrock on which every modern time series "
    "method is built. It combines three simple ideas (autoregression, differencing, "
    "moving average) into something surprisingly powerful, and understanding it means "
    "you will understand every method that came after it. Plus, it is still genuinely "
    "competitive on well-behaved data."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

concept_box(
    "ARIMA(p, d, q) Components",
    "<b>AR(p)</b>: 'The past predicts the future.' The value at time t depends "
    "on the p most recent values. If temperature was high yesterday, it will "
    "probably be high today.<br>"
    "<b>I(d)</b>: 'Make it stationary first.' Difference the series d times to "
    "remove trends before modeling. We covered this in the previous chapter.<br>"
    "<b>MA(q)</b>: 'Learn from your mistakes.' The value at time t depends on "
    "the q most recent *forecast errors*. If you over-predicted yesterday, "
    "adjust down today."
)

formula_box(
    "ARIMA(p, d, q) Model",
    r"\underbrace{Y'_t}_{\text{differenced temp}} = \underbrace{c}_{\text{constant}} + \underbrace{\phi_1 Y'_{t-1} + \cdots + \phi_p Y'_{t-p}}_{\text{AR: weighted recent values}} "
    r"+ \underbrace{\theta_1 \epsilon_{t-1} + \cdots + \theta_q \epsilon_{t-q}}_{\text{MA: weighted recent errors}} + \underbrace{\epsilon_t}_{\text{white noise}}",
    "Y' is the differenced series, phi = AR coefficients ('how much do I trust "
    "yesterday?'), theta = MA coefficients ('how much do I correct for yesterday's "
    "mistake?'), epsilon = white noise. The whole thing is basically a sophisticated "
    "way of saying 'predict from recent values and recent errors.'"
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.subheader("ARIMA Settings")
city = st.sidebar.selectbox("City", CITY_LIST, key="arima_city")
p = st.sidebar.slider("p (AR order)", 0, 5, 2, key="arima_p")
d = st.sidebar.slider("d (Differencing)", 0, 2, 1, key="arima_d")
q = st.sidebar.slider("q (MA order)", 0, 5, 2, key="arima_q")
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 60, 30, key="arima_horizon")

# ── Prepare daily series ─────────────────────────────────────────────────────
city_df = fdf[fdf["city"] == city].copy()
daily = (
    city_df.groupby("date")["temperature_c"]
    .mean()
    .reset_index()
    .sort_values("date")
)
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.set_index("date").asfreq("D")
daily["temperature_c"] = daily["temperature_c"].interpolate(method="linear")

series = daily["temperature_c"]

# Use last portion for validation
n_total = len(series)
n_test = min(forecast_days, n_total // 5)
train = series.iloc[:-n_test]
test = series.iloc[-n_test:]

# ── Section 1: Model Fitting ────────────────────────────────────────────────
st.header(f"1. Fitting ARIMA({p},{d},{q}) on {city}")

st.markdown(
    f"We fit ARIMA({p},{d},{q}) on daily mean temperature, holding out the "
    f"last **{n_test} days** as a test set. The sidebar lets you adjust p, d, "
    "and q -- try different values and watch what happens to the forecast and "
    "the AIC. This kind of hands-on experimentation is worth more than reading "
    "ten textbook chapters."
)

try:
    model = ARIMA(train, order=(p, d, q))
    results = model.fit()

    col1, col2, col3 = st.columns(3)
    col1.metric("AIC", f"{results.aic:.1f}")
    col2.metric("BIC", f"{results.bic:.1f}")
    col3.metric("Log-Likelihood", f"{results.llf:.1f}")

    with st.expander("Model Summary"):
        st.text(str(results.summary()))

    model_fitted = True
except Exception as e:
    st.error(f"Model fitting failed: {e}. Try different (p, d, q) values.")
    model_fitted = False

# ── Section 2: Forecast with Prediction Intervals ───────────────────────────
if model_fitted:
    st.header("2. Forecast with Prediction Intervals")

    forecast = results.get_forecast(steps=n_test)
    fc_mean = forecast.predicted_mean
    fc_ci = forecast.conf_int(alpha=0.05)

    fig_fc = go.Figure()

    # Training data (last 120 days)
    plot_start = max(0, len(train) - 120)
    fig_fc.add_trace(go.Scatter(
        x=train.index[plot_start:], y=train.values[plot_start:],
        mode="lines", name="Training Data",
        line=dict(color="#264653"),
    ))

    # Actual test data
    fig_fc.add_trace(go.Scatter(
        x=test.index, y=test.values,
        mode="lines", name="Actual",
        line=dict(color="#2A9D8F", width=2),
    ))

    # Forecast
    fig_fc.add_trace(go.Scatter(
        x=fc_mean.index, y=fc_mean.values,
        mode="lines", name="Forecast",
        line=dict(color="#E63946", width=2, dash="dash"),
    ))

    # Confidence interval
    fig_fc.add_trace(go.Scatter(
        x=list(fc_ci.index) + list(fc_ci.index[::-1]),
        y=list(fc_ci.iloc[:, 1]) + list(fc_ci.iloc[:, 0][::-1]),
        fill="toself", fillcolor="rgba(230,57,70,0.15)",
        line=dict(width=0), showlegend=True, name="95% CI",
    ))

    apply_common_layout(fig_fc, f"ARIMA({p},{d},{q}) Forecast: {city}", 500)
    st.plotly_chart(fig_fc, use_container_width=True)

    # Forecast accuracy
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(test.values, fc_mean.values[:len(test)])
    rmse = np.sqrt(mean_squared_error(test.values, fc_mean.values[:len(test)]))

    c1, c2 = st.columns(2)
    c1.metric("MAE", f"{mae:.2f} deg C")
    c2.metric("RMSE", f"{rmse:.2f} deg C")

    insight_box(
        "Notice how the prediction interval fans out like a trumpet as we forecast "
        "further ahead. This is ARIMA being honest about its uncertainty -- it knows "
        "that predicting next week is harder than predicting tomorrow. A good model "
        "has narrow intervals that still contain the actual values. If the actual line "
        "wanders outside the shaded region, the model is overconfident."
    )

# ── Section 3: Auto ARIMA ───────────────────────────────────────────────────
st.header("3. Automatic Order Selection (auto_arima)")

st.markdown(
    "Manually tuning p, d, and q is educational but tedious. `pmdarima.auto_arima` "
    "does what you would do if you had infinite patience: it tries a bunch of "
    "different (p, d, q) combinations and picks the one with the lowest AIC. "
    "It is not magic -- it is just a systematic search with some clever shortcuts."
)

try:
    import pmdarima as pm

    with st.spinner("Running auto_arima..."):
        auto_model = pm.auto_arima(
            train, start_p=0, start_q=0,
            max_p=5, max_q=5, d=None,
            seasonal=False, trace=False,
            error_action="ignore", suppress_warnings=True,
            stepwise=True,
        )

    best_order = auto_model.order
    st.success(f"Best order found: ARIMA{best_order} with AIC = {auto_model.aic():.1f}")

    with st.expander("Auto ARIMA Summary"):
        st.text(str(auto_model.summary()))

    # Forecast from auto model
    fc_auto, ci_auto = auto_model.predict(n_periods=n_test, return_conf_int=True, alpha=0.05)

    fig_auto = go.Figure()
    fig_auto.add_trace(go.Scatter(
        x=train.index[plot_start:], y=train.values[plot_start:],
        mode="lines", name="Training", line=dict(color="#264653"),
    ))
    fig_auto.add_trace(go.Scatter(
        x=test.index, y=test.values,
        mode="lines", name="Actual", line=dict(color="#2A9D8F", width=2),
    ))
    fig_auto.add_trace(go.Scatter(
        x=test.index[:len(fc_auto)], y=fc_auto,
        mode="lines", name=f"auto ARIMA{best_order}",
        line=dict(color="#7209B7", width=2, dash="dash"),
    ))
    fig_auto.add_trace(go.Scatter(
        x=list(test.index[:len(fc_auto)]) + list(test.index[:len(fc_auto)][::-1]),
        y=list(ci_auto[:, 1]) + list(ci_auto[:, 0][::-1]),
        fill="toself", fillcolor="rgba(114,9,183,0.15)",
        line=dict(width=0), name="95% CI",
    ))
    apply_common_layout(fig_auto, f"auto_arima Forecast: {city}", 500)
    st.plotly_chart(fig_auto, use_container_width=True)

    mae_auto = mean_absolute_error(test.values[:len(fc_auto)], fc_auto)
    rmse_auto = np.sqrt(mean_squared_error(test.values[:len(fc_auto)], fc_auto))
    c1, c2 = st.columns(2)
    c1.metric("auto_arima MAE", f"{mae_auto:.2f} deg C")
    c2.metric("auto_arima RMSE", f"{rmse_auto:.2f} deg C")

except ImportError:
    st.info(
        "Install `pmdarima` (`pip install pmdarima`) to enable automatic "
        "order selection."
    )

# ── Section 4: Comparing Orders via AIC ──────────────────────────────────────
st.header("4. Comparing Different ARIMA Orders")

st.markdown(
    "Below we fit several ARIMA orders and compare their AIC values. "
    "Lower AIC = better trade-off between goodness-of-fit and complexity. "
    "Think of AIC as a judge who is impressed by accuracy but deeply suspicious "
    "of models that use too many parameters to achieve it."
)

orders_to_try = [
    (1, 1, 0), (1, 1, 1), (2, 1, 1), (2, 1, 2), (3, 1, 1), (0, 1, 1),
]
order_results = []
for o in orders_to_try:
    try:
        m = ARIMA(train, order=o).fit()
        fc_o = m.get_forecast(steps=n_test)
        mae_o = mean_absolute_error(test.values, fc_o.predicted_mean.values[:len(test)])
        order_results.append({
            "Order": f"ARIMA{o}", "AIC": round(m.aic, 1),
            "BIC": round(m.bic, 1), "MAE (test)": round(mae_o, 2),
        })
    except Exception:
        pass

if order_results:
    ord_df = pd.DataFrame(order_results).sort_values("AIC")
    st.dataframe(ord_df, use_container_width=True, hide_index=True)

    fig_aic = go.Figure()
    fig_aic.add_trace(go.Bar(
        x=ord_df["Order"], y=ord_df["AIC"],
        marker_color="#2E86C1",
    ))
    apply_common_layout(fig_aic, "AIC Comparison Across ARIMA Orders", 400)
    st.plotly_chart(fig_aic, use_container_width=True)

    insight_box(
        "Here is the subtle thing about AIC: it penalizes model complexity, so "
        "a higher-order model has to *really* earn its extra parameters through "
        "better fit. The best AIC does not always correspond to the lowest test MAE "
        "(because AIC is estimated on the training set), but it is a surprisingly "
        "reliable heuristic -- especially when you do not have a test set handy."
    )

# ── Section 5: Residual Diagnostics ──────────────────────────────────────────
if model_fitted:
    st.header("5. Residual Diagnostics")

    st.markdown(
        "A well-fitted ARIMA model should have residuals that look like "
        "white noise: mean of zero, constant variance, no autocorrelation. "
        "If the residuals have leftover patterns, the model is leaving money "
        "on the table -- there is predictable structure it failed to capture."
    )

    resids = results.resid.dropna()

    fig_res = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Residuals Over Time", "Residual Distribution"],
        vertical_spacing=0.15,
    )
    fig_res.add_trace(go.Scatter(
        x=resids.index, y=resids.values,
        mode="lines", line=dict(color="#264653", width=0.7), showlegend=False,
    ), row=1, col=1)
    fig_res.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    fig_res.add_trace(go.Histogram(
        x=resids.values, nbinsx=50,
        marker_color="#2E86C1", showlegend=False,
    ), row=2, col=1)
    fig_res.update_layout(height=550, template="plotly_white")
    st.plotly_chart(fig_res, use_container_width=True)

    warning_box(
        "If you see periodicity in the residual plot (a wave that should not be "
        "there) or the histogram looks skewed, the model is missing important "
        "structure. The most common fix for temperature data: use seasonal ARIMA "
        "(SARIMA), which explicitly models the annual cycle. Or jump ahead to "
        "Prophet, which handles multiple seasonalities natively."
    )

code_example("""
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

# Manual ARIMA
model = ARIMA(train, order=(2, 1, 2))
results = model.fit()

# Forecast with prediction intervals
forecast = results.get_forecast(steps=30)
ci = forecast.conf_int(alpha=0.05)

# Automatic order selection
auto = pm.auto_arima(train, seasonal=False, stepwise=True)
print(auto.summary())
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "In ARIMA(2, 1, 1), what does the '1' in the middle represent?",
    [
        "One autoregressive lag",
        "One round of differencing to achieve stationarity",
        "One moving average term",
        "One seasonal period",
    ],
    1,
    "The (p, d, q) notation: p = AR lags, d = differencing order, "
    "q = MA terms. So d=1 means we difference the series once before "
    "fitting the AR and MA parts. It is the 'I' in ARIMA.",
    key="arima_quiz",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "ARIMA combines three ideas: autoregression ('the past predicts the future'), differencing ('make it stationary'), and moving average ('learn from your mistakes').",
    "The d parameter controls differencing; use the ADF test to figure out how much you need. Usually d=1 does the job.",
    "AIC and BIC help select model order; auto_arima automates the search. Think of AIC as a principled way to balance accuracy vs complexity.",
    "Prediction intervals widen with forecast horizon -- this is the model being honest about growing uncertainty. Embrace the trumpet shape.",
    "Always check residuals for leftover patterns. If the residuals are not white noise, the model is telling you it needs to be more complex (or differently structured).",
])
