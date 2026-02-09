"""Chapter 7: Time Series Visualization -- Line charts, resampling, rolling averages."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import line_chart, apply_common_layout, color_map
from utils.constants import (
    CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS, FEATURE_UNITS,
)
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(7, "Time Series Visualization", part="II")
st.markdown(
    "Weather data has a natural superpower that most datasets don't: it's *temporal*. "
    "Every observation has a timestamp, and that timestamp carries meaning. Temperature "
    "at 3am is different from temperature at 3pm for reasons that matter. January is "
    "different from July. This chapter is about how to visualize data that unfolds over "
    "time, and the **surprisingly important decisions** you have to make about resampling "
    "and smoothing."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 7.1 Theory ──────────────────────────────────────────────────────────────
st.header("7.1  Time Series Fundamentals")

concept_box(
    "What Makes Time Series Special",
    "A <b>time series</b> is a sequence of data points indexed by time. The key thing "
    "that makes it different from other data: the order matters. Shuffle a time series "
    "and you've destroyed it. Three things to look for:<br>"
    "- <b>Trend</b>: is the overall level going up, down, or staying flat?<br>"
    "- <b>Seasonality</b>: do you see regular repeating patterns? (Daily, weekly, yearly?)<br>"
    "- <b>Noise</b>: the random fluctuations that make you squint and wonder if that bump "
    "is real or just random variation."
)

concept_box(
    "Resampling: Changing the Clock",
    "<b>Downsampling</b>: going from fine-grained to coarse -- e.g., hourly to daily. "
    "You need to choose an aggregation function (mean, sum, max). This choice matters more "
    "than you'd think: daily mean temperature and daily max temperature tell very different "
    "stories.<br>"
    "<b>Upsampling</b>: going from coarse to fine -- e.g., daily to hourly. Now you need "
    "to interpolate, which means you're literally making up data between the points you have."
)

concept_box(
    "Rolling Averages: The Original Smoothing Technique",
    "A rolling mean (aka moving average) replaces each point with the average of its "
    "neighbors within a window. It smooths out short-term noise to reveal the underlying "
    "trend. The tradeoff: a larger window gives a smoother curve but introduces more lag -- "
    "it responds slowly to real changes. This is the fundamental bias-variance tradeoff, "
    "showing up in one of its simplest forms."
)

# ── 7.2 Interactive line chart with resampling ───────────────────────────────
st.header("7.2  Interactive: Resample & Plot")

col_a, col_b = st.columns(2)
with col_a:
    feat_sel = st.selectbox(
        "Feature", FEATURE_COLS,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
        key="ts_feat",
    )
with col_b:
    resample_freq = st.selectbox(
        "Resample frequency",
        ["Hourly (raw)", "Daily", "Weekly", "Monthly"],
        index=1,
        key="ts_freq",
    )

freq_map = {
    "Hourly (raw)": None,
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "ME",
}
freq = freq_map[resample_freq]

cities_sel = st.multiselect(
    "Cities", sorted(fdf["city"].unique()),
    default=["NYC", "Los Angeles"],
    key="ts_cities",
)

ts_data = fdf[fdf["city"].isin(cities_sel)][["datetime", "city", feat_sel]].dropna()

if len(ts_data) > 0 and cities_sel:
    if freq:
        resampled = (
            ts_data.set_index("datetime")
            .groupby("city")[feat_sel]
            .resample(freq)
            .mean()
            .reset_index()
        )
    else:
        resampled = ts_data.copy()

    fig_ts = px.line(
        resampled, x="datetime", y=feat_sel, color="city",
        color_discrete_map=CITY_COLORS,
        labels={feat_sel: FEATURE_LABELS.get(feat_sel, feat_sel), "datetime": "Date", "city": "City"},
        title=f"{FEATURE_LABELS[feat_sel]} -- {resample_freq}",
    )
    apply_common_layout(fig_ts, height=500)
    st.plotly_chart(fig_ts, use_container_width=True)

    insight_box(
        "Try switching between hourly and monthly. At hourly resolution, the chart is a "
        "chaotic mess -- you can barely tell what's happening. At monthly resolution, the "
        "seasonal pattern jumps out beautifully. The data hasn't changed; only your *resolution* "
        "has. This is a general principle: the right level of aggregation depends on the "
        "pattern you're trying to see."
    )
else:
    st.info("Select at least one city with data available.")

# ── 7.3 Rolling average ─────────────────────────────────────────────────────
st.header("7.3  Rolling Average Overlay")

st.markdown(
    "Here's a nice trick: show the raw daily data in a faint color, and overlay a bold "
    "rolling average on top. This lets you see both the noise and the trend simultaneously. "
    "Adjust the window size to control how much smoothing you get."
)

rolling_window = st.slider(
    "Rolling window size (in resampled periods)",
    min_value=2, max_value=90, value=7, step=1,
    key="ts_rolling_window",
)

if len(ts_data) > 0 and cities_sel:
    # Use daily data for rolling
    daily = (
        ts_data.set_index("datetime")
        .groupby("city")[feat_sel]
        .resample("D")
        .mean()
        .reset_index()
    )

    fig_roll = go.Figure()
    for city in cities_sel:
        cd = daily[daily["city"] == city].sort_values("datetime")
        cd["rolling"] = cd[feat_sel].rolling(rolling_window, min_periods=1).mean()

        fig_roll.add_trace(go.Scatter(
            x=cd["datetime"], y=cd[feat_sel],
            mode="lines", name=f"{city} (daily)",
            line=dict(color=CITY_COLORS.get(city, "grey"), width=0.5),
            opacity=0.3,
        ))
        fig_roll.add_trace(go.Scatter(
            x=cd["datetime"], y=cd["rolling"],
            mode="lines", name=f"{city} ({rolling_window}-day avg)",
            line=dict(color=CITY_COLORS.get(city, "grey"), width=2.5),
        ))

    fig_roll.update_layout(
        xaxis_title="Date",
        yaxis_title=FEATURE_LABELS.get(feat_sel, feat_sel),
    )
    apply_common_layout(fig_roll, title=f"Daily vs {rolling_window}-Day Rolling Average", height=500)
    st.plotly_chart(fig_roll, use_container_width=True)

    warning_box(
        "Rolling averages have an inherent tradeoff you should be aware of: more smoothing "
        "means more lag. A 7-day average responds to changes about 3.5 days late, on average. "
        "A 30-day average is nearly a month behind. If you're trying to detect a sudden cold "
        "snap, a wide rolling window will literally show you the cold snap *after* it's over. "
        "Choose your window based on whether you care more about trend clarity or timing."
    )

# ── 7.4 Diurnal pattern ─────────────────────────────────────────────────────
st.header("7.4  Diurnal (Daily) Pattern")

st.markdown(
    "Here's something satisfying: temperature follows a beautifully predictable daily "
    "cycle. Cool at night, warm during the day, peak in the afternoon, cool again. "
    "We can see this by averaging each hour across all days. The result is a clean "
    "signal that emerges from thousands of noisy individual days."
)

diurnal_city = st.selectbox(
    "City for diurnal pattern", sorted(fdf["city"].unique()),
    key="ts_diurnal_city",
)

diurnal = (
    fdf[fdf["city"] == diurnal_city]
    .groupby("hour")[FEATURE_COLS]
    .mean()
    .reset_index()
)

if len(diurnal) > 0:
    fig_diurnal = px.line(
        diurnal, x="hour", y=feat_sel,
        labels={"hour": "Hour of Day", feat_sel: FEATURE_LABELS.get(feat_sel, feat_sel)},
        title=f"Average {FEATURE_LABELS[feat_sel]} by Hour -- {diurnal_city}",
        markers=True,
    )
    fig_diurnal.update_traces(line_color=CITY_COLORS.get(diurnal_city, "steelblue"))
    apply_common_layout(fig_diurnal, height=400)
    st.plotly_chart(fig_diurnal, use_container_width=True)

# ── 7.5 NYC vs LA comparison ────────────────────────────────────────────────
st.header("7.5  Case Study: NYC vs Los Angeles")

insight_box(
    "This is perhaps the single most illustrative comparison in the whole dataset. NYC "
    "has dramatic seasonality -- it's genuinely cold in winter and hot in summer, with a "
    "massive annual range. LA barely moves. It's mild in January and mild in August. "
    "Overlaying their monthly time series makes this contrast visceral in a way that "
    "no summary statistic can."
)

nyc_la = fdf[fdf["city"].isin(["NYC", "Los Angeles"])][["datetime", "city", "temperature_c"]].dropna()
if len(nyc_la) > 0:
    nyc_la_monthly = (
        nyc_la.set_index("datetime")
        .groupby("city")["temperature_c"]
        .resample("ME")
        .mean()
        .reset_index()
    )
    fig_nycla = px.line(
        nyc_la_monthly, x="datetime", y="temperature_c", color="city",
        color_discrete_map=CITY_COLORS,
        labels={"temperature_c": "Temperature (°C)", "datetime": "Date"},
        title="Monthly Avg Temperature: NYC vs Los Angeles",
    )
    apply_common_layout(fig_nycla, height=450)
    st.plotly_chart(fig_nycla, use_container_width=True)
else:
    st.info("NYC or Los Angeles data not available in current filter.")

# ── Code example ─────────────────────────────────────────────────────────────
code_example(
    """import pandas as pd
import plotly.express as px

# Set datetime index
ts = df.set_index("datetime")

# Resample to daily mean
daily = ts.groupby("city")["temperature_c"].resample("D").mean().reset_index()

# Rolling average
daily["rolling_7d"] = (
    daily.groupby("city")["temperature_c"]
    .transform(lambda s: s.rolling(7, min_periods=1).mean())
)

# Line chart
fig = px.line(daily, x="datetime", y="rolling_7d", color="city")
fig.show()

# Diurnal pattern
diurnal = df.groupby("hour")["temperature_c"].mean()
diurnal.plot(title="Average Temperature by Hour of Day")
"""
)

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "Resampling from hourly to daily using .resample('D').mean() is an example of:",
    [
        "Upsampling",
        "Downsampling",
        "Interpolation",
        "Normalization",
    ],
    correct_idx=1,
    explanation="Going from a finer frequency (hourly) to a coarser one (daily) is downsampling. "
                "You're throwing away temporal resolution in exchange for a cleaner signal. "
                "The .mean() tells pandas how to aggregate the multiple hourly values into one daily value.",
    key="ch7_quiz1",
)

quiz(
    "A larger rolling window produces a curve that is:",
    [
        "More noisy",
        "Smoother but with more lag",
        "Identical to the original",
        "Shifted to the right on the y-axis",
    ],
    correct_idx=1,
    explanation="A larger window averages more data points, which smooths out noise but makes "
                "the curve respond sluggishly to real changes. It's the classic smoothness-vs-responsiveness "
                "tradeoff, and there's no free lunch here.",
    key="ch7_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "Line charts are the default for time series. Time goes on the x-axis, always. Breaking this convention confuses everyone.",
    "Downsampling (hourly to daily, daily to monthly) reduces noise and makes patterns visible. Choose the resolution that matches your question.",
    "Rolling averages reveal trends but introduce lag proportional to the window size. There's no free lunch -- smoothness costs responsiveness.",
    "Weather data shows beautiful nested cycles: diurnal (24-hour) patterns inside seasonal (yearly) patterns. Both are real; which you see depends on your resolution.",
    "Comparing cities on the same axes is one of the most effective ways to reveal differences in variability and seasonality.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 6: Scatter Plots & Relationships",
    prev_page="06_Scatter_Plots_and_Relationships.py",
    next_label="Ch 8: Box Plots & Comparisons",
    next_page="08_Box_Plots_and_Comparisons.py",
)
