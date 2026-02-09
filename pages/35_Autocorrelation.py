"""Chapter 35 -- Autocorrelation, ACF, PACF, and Stationarity."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(35, "Autocorrelation & Stationarity", part="VIII")

st.markdown(
    "Here is a question that sounds simple but turns out to be deeply important: "
    "how much does today's temperature tell you about tomorrow's? What about "
    "yesterday's temperature -- does it predict today? Autocorrelation formalizes "
    "this intuition: it measures how correlated a time series is with lagged copies "
    "of itself. It is the single most useful diagnostic tool in all of time series "
    "analysis, and understanding it properly is the difference between building "
    "forecasting models that work and ones that are basically random number generators "
    "with extra steps."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

concept_box(
    "Autocorrelation Function (ACF)",
    "The ACF at lag k is just the Pearson correlation between Y_t and Y_{t-k}. "
    "If the ACF at lag 24 is high, it means the temperature now is strongly "
    "correlated with the temperature 24 time steps ago. A slow, linear decay "
    "in the ACF is the telltale sign of a trend. Periodic spikes are the "
    "telltale sign of seasonality. These patterns are so distinctive that an "
    "experienced time series analyst can diagnose a series just by glancing at "
    "its ACF plot -- it is like reading an EKG for data."
)

formula_box(
    "Autocorrelation at Lag k",
    r"\underbrace{\rho_k}_{\text{autocorrelation at lag k}} = \frac{\underbrace{\text{Cov}(Y_t, Y_{t-k})}_{\text{covariance with lagged self}}}{\underbrace{\text{Var}(Y_t)}_{\text{overall variance}}}",
    "rho_k ranges from -1 to 1. Values near 0 mean no linear dependence at that "
    "lag. Values near 1 mean 'knowing the value k steps ago basically tells you "
    "the value now.'"
)

concept_box(
    "Partial Autocorrelation Function (PACF)",
    "The PACF is the trickier sibling. The ACF at lag 3 includes both the "
    "direct effect of lag 3 *and* the indirect effects transmitted through "
    "lags 1 and 2. The PACF strips away the intermediaries -- it measures the "
    "correlation between Y_t and Y_{t-k} <b>after removing the effect of all "
    "intervening lags</b>. This is crucial for model selection: if the PACF "
    "cuts off sharply after p lags, you probably want an AR(p) model."
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.subheader("ACF Settings")
city = st.sidebar.selectbox("City", CITY_LIST, key="acf_city")
max_lag = st.sidebar.slider("Max Lag (hours)", 24, 168, 72, 12, key="acf_maxlag")
use_daily = st.sidebar.checkbox("Use Daily Averages (instead of hourly)", value=False, key="acf_daily")

# ── Prepare series ───────────────────────────────────────────────────────────
city_df = fdf[fdf["city"] == city].sort_values("datetime").copy()

if use_daily:
    series_df = city_df.groupby("date")["temperature_c"].mean().reset_index()
    series_df["date"] = pd.to_datetime(series_df["date"])
    series_df = series_df.set_index("date").sort_index()
    series = series_df["temperature_c"].dropna()
    time_label = "days"
else:
    series = city_df.set_index("datetime")["temperature_c"].dropna()
    time_label = "hours"

# ── Section 1: Raw Series ────────────────────────────────────────────────────
st.header(f"1. Temperature Time Series -- {city}")

fig_raw = go.Figure()
fig_raw.add_trace(go.Scatter(
    x=series.index, y=series.values,
    mode="lines", name="Temperature",
    line=dict(color=CITY_COLORS.get(city, "#2E86C1"), width=1),
))
apply_common_layout(fig_raw, f"Temperature ({time_label}): {city}", 350)
st.plotly_chart(fig_raw, use_container_width=True)

# ── Section 2: ACF and PACF ─────────────────────────────────────────────────
st.header("2. ACF and PACF Plots")

st.markdown(
    f"We compute autocorrelation up to **{max_lag} {time_label}** lag. "
    "The red dashed lines mark the 95% confidence interval -- any bar that "
    "pokes above (or below) those lines is statistically significant. "
    "What you are looking for: periodic spikes that betray cyclical patterns."
)

acf_vals = acf(series, nlags=max_lag, fft=True)
pacf_vals = pacf(series, nlags=min(max_lag, len(series) // 2 - 1))
n = len(series)
ci = 1.96 / np.sqrt(n)

# ACF plot
fig_acf = make_subplots(rows=2, cols=1, subplot_titles=["ACF", "PACF"],
                        shared_xaxes=True, vertical_spacing=0.12)

lags = list(range(len(acf_vals)))
fig_acf.add_trace(go.Bar(
    x=lags, y=acf_vals, marker_color="#2E86C1", showlegend=False,
), row=1, col=1)
fig_acf.add_hline(y=ci, line_dash="dash", line_color="red", row=1, col=1)
fig_acf.add_hline(y=-ci, line_dash="dash", line_color="red", row=1, col=1)

pacf_lags = list(range(len(pacf_vals)))
fig_acf.add_trace(go.Bar(
    x=pacf_lags, y=pacf_vals, marker_color="#2A9D8F", showlegend=False,
), row=2, col=1)
fig_acf.add_hline(y=ci, line_dash="dash", line_color="red", row=2, col=1)
fig_acf.add_hline(y=-ci, line_dash="dash", line_color="red", row=2, col=1)

fig_acf.update_xaxes(title_text=f"Lag ({time_label})", row=2, col=1)
fig_acf.update_yaxes(title_text="ACF", row=1, col=1)
fig_acf.update_yaxes(title_text="PACF", row=2, col=1)
fig_acf.update_layout(height=550, template="plotly_white", margin=dict(t=40, b=40))
st.plotly_chart(fig_acf, use_container_width=True)

if not use_daily:
    insight_box(
        "The truly beautiful thing about the ACF plot is the **spike at lag 24**. "
        "That is the diurnal cycle, plain as day (pun intended). It means that "
        "temperature at 2 PM today is highly correlated with temperature at 2 PM "
        "yesterday, which makes complete physical sense -- the sun does roughly the "
        "same thing every 24 hours. The PACF shows which lags have *direct* influence "
        "after controlling for the intermediate hours, which is the information you "
        "actually need for building an AR model."
    )
else:
    insight_box(
        "With daily data, the ACF shows a slow, graceful decay -- warm days follow "
        "warm days, cold follows cold. This is the strong persistence of temperature: "
        "weather has momentum. If it was 35 degrees today, it is probably not going to "
        "be 5 degrees tomorrow. There may also be a subtle annual cycle visible as a "
        "broad hump in the ACF."
    )

code_example("""
from statsmodels.tsa.stattools import acf, pacf

acf_values = acf(series, nlags=72, fft=True)
pacf_values = pacf(series, nlags=72)

# Confidence interval: +/- 1.96 / sqrt(n)
ci = 1.96 / np.sqrt(len(series))
""")

# ── Section 3: Stationarity & ADF Test ───────────────────────────────────────
st.header("3. Stationarity & Augmented Dickey-Fuller Test")

concept_box(
    "Stationarity",
    "A time series is <b>stationary</b> if its statistical properties -- mean, "
    "variance, autocorrelation structure -- do not change over time. This is the "
    "key assumption underlying most forecasting models (ARIMA, etc.). The "
    "<b>Augmented Dickey-Fuller (ADF)</b> test is the standard way to check. "
    "It works like a court trial:<br>"
    "H0 (the null hypothesis, the 'defendant'): The series has a unit root (non-stationary).<br>"
    "H1 (the alternative): The series is stationary.<br>"
    "A low p-value means 'guilty of being stationary' -- which is actually what you want."
)

adf_result = adfuller(series.values[:min(5000, len(series))], autolag="AIC")

col1, col2, col3 = st.columns(3)
col1.metric("ADF Statistic", f"{adf_result[0]:.4f}")
col2.metric("p-value", f"{adf_result[1]:.4f}")
col3.metric("Lags Used", adf_result[2])

if adf_result[1] < 0.05:
    st.success(
        f"p-value = {adf_result[1]:.4f} < 0.05: We **reject** H0 -- the "
        f"series appears stationary (or trend-stationary)."
    )
else:
    st.error(
        f"p-value = {adf_result[1]:.4f} >= 0.05: We **fail to reject** H0 -- "
        f"the series may be non-stationary. Differencing may help."
    )

st.markdown("**Critical values:**")
crit_df = pd.DataFrame({
    "Significance Level": ["1%", "5%", "10%"],
    "Critical Value": [adf_result[4]["1%"], adf_result[4]["5%"], adf_result[4]["10%"]],
}).round(4)
st.dataframe(crit_df, use_container_width=True, hide_index=True)

# ── Section 4: Differencing ──────────────────────────────────────────────────
st.header("4. Differencing to Achieve Stationarity")

st.markdown(
    "If your series is non-stationary (or you want to be safe), the standard cure "
    "is **differencing**: you subtract the previous value to get the change. "
    "Instead of looking at 'what is the temperature?', you look at 'how much did "
    "the temperature *change*?' This removes trends and is literally the 'd' in "
    "ARIMA(p, d, q). It is such a common operation that it has its own notation: "
    "`Y'_t = Y_t - Y_{t-1}`."
)

diff1 = series.diff().dropna()

fig_diff = make_subplots(
    rows=2, cols=1,
    subplot_titles=["Original Series", "After 1st Differencing"],
    shared_xaxes=True, vertical_spacing=0.1,
)
fig_diff.add_trace(go.Scatter(
    x=series.index[:2000], y=series.values[:2000],
    mode="lines", line=dict(color="#264653"), showlegend=False,
), row=1, col=1)
fig_diff.add_trace(go.Scatter(
    x=diff1.index[:2000], y=diff1.values[:2000],
    mode="lines", line=dict(color="#E63946"), showlegend=False,
), row=2, col=1)
fig_diff.update_layout(height=500, template="plotly_white")
st.plotly_chart(fig_diff, use_container_width=True)

# ADF on differenced
adf_diff = adfuller(diff1.values[:min(5000, len(diff1))], autolag="AIC")
c1, c2 = st.columns(2)
c1.metric("ADF Stat (Original)", f"{adf_result[0]:.4f}")
c2.metric("ADF Stat (Differenced)", f"{adf_diff[0]:.4f}")

st.markdown(
    f"After differencing, the ADF statistic becomes **{adf_diff[0]:.2f}** "
    f"(p = {adf_diff[1]:.4f}), which is extremely stationary. The differenced "
    "series oscillates around zero with no trend -- exactly what ARIMA wants to see."
)

# ACF of differenced series
st.subheader("ACF of Differenced Series")
acf_diff = acf(diff1.values[:min(5000, len(diff1))], nlags=max_lag, fft=True)
fig_acf_d = go.Figure()
fig_acf_d.add_trace(go.Bar(
    x=list(range(len(acf_diff))), y=acf_diff,
    marker_color="#E63946",
))
fig_acf_d.add_hline(y=1.96/np.sqrt(len(diff1)), line_dash="dash", line_color="gray")
fig_acf_d.add_hline(y=-1.96/np.sqrt(len(diff1)), line_dash="dash", line_color="gray")
fig_acf_d.update_layout(xaxis_title=f"Lag ({time_label})", yaxis_title="ACF")
apply_common_layout(fig_acf_d, "ACF of Differenced Series", 350)
st.plotly_chart(fig_acf_d, use_container_width=True)

insight_box(
    "Differencing kills the trend, but notice that the periodic spikes (lag 24 for "
    "hourly data) often survive -- they represent seasonal autocorrelation, which is "
    "a different beast. To handle those, you would need seasonal differencing or a "
    "SARIMA model. It is a bit like pulling a weed and discovering the root goes deeper "
    "than you expected."
)

warning_box(
    "Over-differencing is a real danger. Each round of differencing can introduce "
    "artificial patterns (specifically, it adds a unit root in the MA component). "
    "Usually d=1 is enough. If you find yourself reaching for d=2, stop and ask "
    "whether the original series was really that non-stationary, or whether something "
    "else is going on."
)

code_example("""
from statsmodels.tsa.stattools import adfuller

# ADF test
result = adfuller(series, autolag='AIC')
print(f'ADF Stat: {result[0]:.4f}, p-value: {result[1]:.4f}')

# First difference
diff_series = series.diff().dropna()
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "In hourly weather data, you see a strong ACF spike at lag 24. "
    "What does this indicate?",
    [
        "There is a 24-month cycle",
        "There is a diurnal (daily) cycle -- temperature repeats every 24 hours",
        "The data has 24 outliers",
        "The first 24 observations are missing",
    ],
    1,
    "Lag 24 in hourly data corresponds to exactly one day. The spike means "
    "temperature at 2 PM today is highly correlated with temperature at 2 PM "
    "yesterday. The sun is remarkably consistent in its habits.",
    key="acf_quiz",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "ACF measures correlation between a series and its lagged self; PACF strips out the effect of intermediate lags. Together, they are the EKG of your time series.",
    "Periodic ACF spikes reveal cycles (lag 24 = diurnal for hourly data, lag 365 = annual for daily data). These patterns are diagnostic gold.",
    "The ADF test checks for unit roots (non-stationarity); a small p-value means stationary, which is what you want for most forecasting models.",
    "Differencing removes trends and is the 'd' in ARIMA(p, d, q). It is simple, effective, and the first thing you try when stationarity fails.",
    "Always check both ACF and PACF to guide your choice of AR and MA orders. The ACF tells you the overall story; the PACF tells you the direct effects.",
])
