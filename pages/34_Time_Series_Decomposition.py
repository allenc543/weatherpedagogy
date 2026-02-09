"""Chapter 34 -- Time Series Decomposition."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL, seasonal_decompose

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, line_chart
from utils.constants import CITY_COLORS, CITY_LIST
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(34, "Time Series Decomposition", part="VIII")

st.markdown(
    "Here is the basic idea behind time series decomposition, and it is one of those "
    "ideas that seems almost too simple to be useful, and then turns out to be "
    "enormously useful. You take a time series and say: 'This signal is made of "
    "three things added together (or multiplied together) -- a long-term **trend** "
    "(is it getting warmer over the years?), a repeating **seasonal** pattern "
    "(summer is hot, winter is cold), and a **residual** (the unpredictable stuff "
    "that is left over).' Decomposition is the act of pulling these apart so you "
    "can study each one in isolation. It is like separating a song into vocals, "
    "guitar, and drums."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

concept_box(
    "Components of a Time Series",
    "<b>Trend</b>: The slow-moving direction of the series. Is it going up, down, "
    "or nowhere over the long haul?<br>"
    "<b>Seasonality</b>: The repeating patterns that occur at known intervals. "
    "For daily temperature, this is the annual warm-cold cycle. It is predictable "
    "by definition.<br>"
    "<b>Residual</b>: Everything left over after you remove trend and seasonality. "
    "In an ideal world, this is just random noise. In practice, it is also where "
    "any structure you missed lives."
)

formula_box(
    "Additive Decomposition",
    r"\underbrace{Y_t}_{\text{observed temp}} = \underbrace{T_t}_{\text{long-term trend}} + \underbrace{S_t}_{\text{seasonal pattern}} + \underbrace{R_t}_{\text{residual noise}}",
    "You use this when the seasonal swings stay roughly the same size regardless "
    "of the level. Temperature data almost always works this way -- a 20-degree "
    "summer-winter swing is a 20-degree swing whether the baseline is 15 or 18."
)

formula_box(
    "Multiplicative Decomposition",
    r"\underbrace{Y_t}_{\text{observed value}} = \underbrace{T_t}_{\text{long-term trend}} \times \underbrace{S_t}_{\text{seasonal multiplier}} \times \underbrace{R_t}_{\text{residual factor}}",
    "You use this when the seasonal swings grow proportionally with the trend. "
    "Think of monthly ice cream sales as the overall market grows -- the peaks "
    "get taller as the baseline rises."
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.subheader("Decomposition Settings")
city = st.sidebar.selectbox("City", CITY_LIST, key="decomp_city")
model_type = st.sidebar.radio(
    "Decomposition Type",
    ["additive", "multiplicative"],
    key="decomp_model",
)

# ── Prepare daily mean temperature ───────────────────────────────────────────
city_df = fdf[fdf["city"] == city].copy()
daily = (
    city_df.groupby("date")["temperature_c"]
    .mean()
    .reset_index()
    .sort_values("date")
)
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.set_index("date")
daily = daily.asfreq("D")
daily["temperature_c"] = daily["temperature_c"].interpolate(method="linear")

# ── Section 1: Raw Series ────────────────────────────────────────────────────
st.header(f"1. Daily Mean Temperature -- {city}")

fig_raw = go.Figure()
fig_raw.add_trace(go.Scatter(
    x=daily.index, y=daily["temperature_c"],
    mode="lines", name="Daily Mean Temp",
    line=dict(color=CITY_COLORS.get(city, "#2E86C1")),
))
apply_common_layout(fig_raw, f"Daily Mean Temperature: {city}", 400)
st.plotly_chart(fig_raw, use_container_width=True)

st.markdown(
    "Even from a quick glance, you can see the annual cycle -- the data breathes in "
    "and out like a wave, warm summers and cooler winters. Decomposition is going to "
    "tell us something more precise: exactly *how big* that seasonal swing is, whether "
    "there is a gradual trend underneath it, and how much of the day-to-day variation "
    "is just noise."
)

# ── Section 2: Classical Decomposition ───────────────────────────────────────
st.header("2. Classical Decomposition")

st.markdown(
    f"We are using **{model_type}** decomposition with a period of 365 days -- "
    "i.e., we are telling the algorithm 'there is an annual cycle, please find it.' "
    "You can toggle between additive and multiplicative in the sidebar, though for "
    "temperature data the additive version is almost always the right call."
)

if model_type == "multiplicative" and (daily["temperature_c"] <= 0).any():
    st.warning(
        "Multiplicative decomposition requires strictly positive values. "
        "Shifting temperatures above zero for demonstration."
    )
    series = daily["temperature_c"] - daily["temperature_c"].min() + 1
else:
    series = daily["temperature_c"]

try:
    decomp = seasonal_decompose(series, model=model_type, period=365)

    fig_decomp = make_subplots(
        rows=4, cols=1,
        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
        shared_xaxes=True, vertical_spacing=0.06,
    )
    components = [
        (decomp.observed, "#264653"),
        (decomp.trend, "#E63946"),
        (decomp.seasonal, "#2A9D8F"),
        (decomp.resid, "#F4A261"),
    ]
    for i, (comp, color) in enumerate(components, 1):
        fig_decomp.add_trace(
            go.Scatter(x=comp.index, y=comp.values, mode="lines",
                       line=dict(color=color), showlegend=False),
            row=i, col=1,
        )
    fig_decomp.update_layout(
        height=700, template="plotly_white",
        margin=dict(t=40, b=30, l=60, r=30),
    )
    st.plotly_chart(fig_decomp, use_container_width=True)

except Exception as e:
    st.error(f"Decomposition failed: {e}")

insight_box(
    "The **seasonal** component shows a beautiful, almost sinusoidal annual pattern -- "
    "this is what scientists call 'seasons.' The **trend** captures any gradual warming "
    "or cooling over the data period. And the **residual** is the leftover jitter -- "
    "ideally close to white noise, meaning the decomposition captured everything "
    "systematic. If your residuals have obvious patterns, the decomposition missed "
    "something."
)

# ── Section 3: STL Decomposition ─────────────────────────────────────────────
st.header("3. STL Decomposition (Robust)")

st.markdown(
    "Classical decomposition has a certain charming naivete: it assumes the seasonal "
    "pattern is *exactly* the same every year. Real weather does not work that way. "
    "**STL** (Seasonal and Trend decomposition using Loess) is more sophisticated -- "
    "it allows the seasonal component to evolve over time, handles arbitrary "
    "seasonality lengths, and is robust to outliers. If classical decomposition is "
    "the hand calculator, STL is the spreadsheet."
)

try:
    stl = STL(daily["temperature_c"], period=365, robust=True)
    stl_result = stl.fit()

    fig_stl = make_subplots(
        rows=4, cols=1,
        subplot_titles=["Observed", "Trend (STL)", "Seasonal (STL)", "Residual (STL)"],
        shared_xaxes=True, vertical_spacing=0.06,
    )
    stl_components = [
        (stl_result.observed, "#264653"),
        (stl_result.trend, "#E63946"),
        (stl_result.seasonal, "#2A9D8F"),
        (stl_result.resid, "#F4A261"),
    ]
    for i, (comp, color) in enumerate(stl_components, 1):
        fig_stl.add_trace(
            go.Scatter(x=comp.index, y=comp.values, mode="lines",
                       line=dict(color=color), showlegend=False),
            row=i, col=1,
        )
    fig_stl.update_layout(
        height=700, template="plotly_white",
        margin=dict(t=40, b=30, l=60, r=30),
    )
    st.plotly_chart(fig_stl, use_container_width=True)

except Exception as e:
    st.error(f"STL decomposition failed: {e}")

code_example("""
from statsmodels.tsa.seasonal import STL, seasonal_decompose

# Classical decomposition
result = seasonal_decompose(series, model='additive', period=365)

# STL (more robust)
stl = STL(series, period=365, robust=True)
stl_result = stl.fit()

# Access components
trend = stl_result.trend
seasonal = stl_result.seasonal
residual = stl_result.resid
""")

# ── Section 4: Comparing Cities ──────────────────────────────────────────────
st.header("4. Seasonal Component Across Cities")

st.markdown(
    "The truly interesting thing happens when you extract the seasonal component "
    "for each city and overlay them. Different climates produce measurably different "
    "seasonal signatures -- and this is where you go from 'decomposition is a neat "
    "mathematical trick' to 'decomposition tells me something real about the world.'"
)

seasonal_df = pd.DataFrame()
for c in CITY_LIST:
    cdf = df[df["city"] == c].groupby("date")["temperature_c"].mean().reset_index()
    cdf["date"] = pd.to_datetime(cdf["date"])
    cdf = cdf.set_index("date").sort_index().asfreq("D")
    cdf["temperature_c"] = cdf["temperature_c"].interpolate()
    if len(cdf) > 365:
        try:
            s = STL(cdf["temperature_c"], period=365, robust=True).fit()
            sdf = pd.DataFrame({"date": s.seasonal.index, "seasonal": s.seasonal.values, "city": c})
            seasonal_df = pd.concat([seasonal_df, sdf], ignore_index=True)
        except Exception:
            pass

if not seasonal_df.empty:
    fig_all = go.Figure()
    for c in CITY_LIST:
        cdata = seasonal_df[seasonal_df["city"] == c]
        if not cdata.empty:
            fig_all.add_trace(go.Scatter(
                x=cdata["date"], y=cdata["seasonal"],
                mode="lines", name=c,
                line=dict(color=CITY_COLORS.get(c, None)),
            ))
    apply_common_layout(fig_all, "Annual Seasonal Component by City", 450)
    fig_all.update_layout(yaxis_title="Seasonal Component (deg C)")
    st.plotly_chart(fig_all, use_container_width=True)

    insight_box(
        "LA has the smallest seasonal amplitude -- it barely notices what time of "
        "year it is. This is what maritime climate does for you. Meanwhile, Dallas "
        "and NYC swing wildly between scorching summers and frigid winters, because "
        "continental climates are nothing if not dramatic. These amplitude differences "
        "are a genuine, quantifiable characteristic of each climate."
    )

warning_box(
    "Classical decomposition assumes the seasonal pattern is perfectly repeating -- "
    "every year, exactly the same. For real weather data, this assumption is wrong "
    "in ways that matter: some winters are milder than others, some summers break "
    "records. STL is preferred because it lets the seasonal component evolve, which "
    "is closer to how the atmosphere actually works."
)

# ── Section 5: Additive vs Multiplicative ────────────────────────────────────
st.header("5. Additive vs Multiplicative -- When to Use Which")

st.markdown("""
| **Property** | **Additive** | **Multiplicative** |
|---|---|---|
| Seasonal amplitude | Constant over time | Proportional to trend |
| Formula | Y = T + S + R | Y = T x S x R |
| Temperature data | Usually appropriate | Rarely needed |
| Sales data with growth | Not ideal | Often better |
""")

st.markdown(
    "For our weather data, additive decomposition is almost always the right choice. "
    "The summer-winter swing does not get bigger just because the average temperature "
    "is a degree higher this decade. Multiplicative decomposition is designed for "
    "things like revenue, where seasonal peaks scale with the overall level -- "
    "Christmas sales get bigger every year because the *total* market is growing."
)

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "If a city's summer-winter temperature difference is roughly the same "
    "every year regardless of the average temperature, which decomposition "
    "model is appropriate?",
    [
        "Multiplicative",
        "Additive",
        "Neither -- use differencing instead",
        "Both give identical results",
    ],
    1,
    "Constant seasonal amplitude = additive model. If the swings grow proportionally "
    "with the level, that is when you reach for multiplicative. For temperature, "
    "additive is almost always correct.",
    key="decomp_quiz",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "Time series = Trend + Seasonality + Residual (additive) or Trend x Seasonality x Residual (multiplicative). This decomposition is one of the most useful 'simple' ideas in all of data analysis.",
    "Classical decomposition assumes a rigid seasonal pattern; STL is more flexible, robust to outliers, and generally preferred for real-world data.",
    "Weather temperature shows a clear 365-day seasonal cycle, which is what scientists call 'seasons.'",
    "LA has the smallest seasonal amplitude; Dallas and NYC the largest -- a direct fingerprint of maritime vs continental climate.",
    "Decomposition is a key first step before forecasting. It tells you *what patterns exist* so you know what your model needs to capture.",
])
