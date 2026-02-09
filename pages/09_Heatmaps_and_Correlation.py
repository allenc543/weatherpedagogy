"""Chapter 9: Heatmaps & Correlation Matrices -- Correlation, calendar heatmaps, pivot tables."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import heatmap_chart, apply_common_layout, color_map
from utils.constants import (
    CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS, FEATURE_UNITS,
)
from utils.stats_helpers import correlation_matrix
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(9, "Heatmaps & Correlation Matrices", part="II")
st.markdown(
    "Here's a problem: you've got a matrix of numbers -- maybe correlations between "
    "every pair of weather features, or average temperatures for every hour-of-day and "
    "month combination. How do you look at a 12x24 grid of numbers without your eyes "
    "glazing over? The answer is heatmaps: they map numbers to colors, and it turns out "
    "your visual system is **absurdly good at spotting color patterns**, much better than "
    "scanning rows of digits."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 9.1 Theory ──────────────────────────────────────────────────────────────
st.header("9.1  Heatmap Fundamentals")

concept_box(
    "When to Reach for a Heatmap",
    "Heatmaps shine when you have a <b>matrix</b> of numbers and want to spot high/low "
    "regions at a glance. Three classic use cases:<br>"
    "- <b>Correlation matrices</b>: which pairs of variables move together?<br>"
    "- <b>Pivot tables</b>: e.g., average temperature for every combination of hour and "
    "month. This is where you see both daily and seasonal patterns in a single image.<br>"
    "- <b>Calendar heatmaps</b>: daily values arranged by week, good for spotting anomalies."
)

concept_box(
    "Correlation Matrices: A Map of Relationships",
    "A correlation matrix shows the Pearson (or Spearman) correlation between every pair "
    "of numeric variables. It's always symmetric (the correlation of A with B equals B with A) "
    "and the diagonal is always 1 (everything correlates perfectly with itself, which is not "
    "very informative but mathematically necessary). The real action is in the off-diagonal "
    "cells -- that's where you find out which variables are friends and which are enemies."
)

# ── 9.2 Hour x Month heatmap ────────────────────────────────────────────────
st.header("9.2  Hour x Month Temperature Heatmap")

st.markdown(
    "This is one of my favorite visualizations in the entire course. It takes two "
    "dimensions of time -- hour of day and month of year -- and shows you the average "
    "temperature for each combination. You get to see the **diurnal cycle** (hot "
    "afternoons, cool nights) and the **seasonal cycle** (cold winters, hot summers) "
    "simultaneously, in a single image. Select a city and take a look."
)

city_sel = st.selectbox("City", sorted(fdf["city"].unique()), key="hm_city")

city_data = fdf[fdf["city"] == city_sel]

# Pivot: rows = hour, columns = month
pivot_temp = city_data.pivot_table(
    index="hour", columns="month", values="temperature_c", aggfunc="mean"
)

# Relabel columns to month names
month_names = {m: pd.Timestamp(2024, m, 1).strftime("%b") for m in range(1, 13)}
pivot_temp = pivot_temp.rename(columns=month_names)

if not pivot_temp.empty:
    fig_hm = go.Figure(data=go.Heatmap(
        z=pivot_temp.values,
        x=pivot_temp.columns.tolist(),
        y=pivot_temp.index.tolist(),
        colorscale="RdYlBu_r",
        colorbar=dict(title="°C"),
    ))
    fig_hm.update_layout(
        xaxis_title="Month",
        yaxis_title="Hour of Day",
        yaxis=dict(autorange="reversed"),
    )
    apply_common_layout(fig_hm, title=f"{city_sel}: Average Temperature by Hour & Month", height=500)
    st.plotly_chart(fig_hm, use_container_width=True)

    insight_box(
        "The hottest cell is summer afternoons (bottom-right area). The coolest is winter "
        "mornings (top-left). But what's really satisfying is how the gradient transitions "
        "smoothly -- there aren't jarring discontinuities, because temperature is a continuous "
        "physical process. An entire year of diurnal and seasonal patterns, compressed into "
        "one image. This is what good visualization does."
    )

# ── 9.3 Other features heatmap ──────────────────────────────────────────────
st.header("9.3  Hour x Month Heatmap for Other Features")

feat_hm = st.selectbox(
    "Feature",
    FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    key="hm_feat",
)

pivot_feat = city_data.pivot_table(
    index="hour", columns="month", values=feat_hm, aggfunc="mean"
)
pivot_feat = pivot_feat.rename(columns=month_names)

if not pivot_feat.empty:
    # Choose color scale based on feature
    cscale = "RdYlBu_r" if feat_hm == "temperature_c" else "Viridis"
    fig_feat_hm = go.Figure(data=go.Heatmap(
        z=pivot_feat.values,
        x=pivot_feat.columns.tolist(),
        y=pivot_feat.index.tolist(),
        colorscale=cscale,
        colorbar=dict(title=FEATURE_UNITS.get(feat_hm, "")),
    ))
    fig_feat_hm.update_layout(
        xaxis_title="Month",
        yaxis_title="Hour of Day",
        yaxis=dict(autorange="reversed"),
    )
    apply_common_layout(fig_feat_hm,
                        title=f"{city_sel}: {FEATURE_LABELS[feat_hm]} by Hour & Month",
                        height=500)
    st.plotly_chart(fig_feat_hm, use_container_width=True)

# ── 9.4 Correlation Matrix ──────────────────────────────────────────────────
st.header("9.4  Correlation Matrix")

corr_method = st.radio(
    "Correlation method",
    ["pearson", "spearman"],
    horizontal=True,
    key="hm_corr_method",
)

st.subheader(f"Per-City Feature Correlation ({corr_method.title()})")

city_for_corr = st.selectbox(
    "City for correlation", sorted(fdf["city"].unique()), key="hm_corr_city"
)

corr_data = fdf[fdf["city"] == city_for_corr][FEATURE_COLS].dropna()
if len(corr_data) > 2:
    corr = corr_data.corr(method=corr_method).round(3)
    corr_labeled = corr.rename(index=FEATURE_LABELS, columns=FEATURE_LABELS)

    # Annotated heatmap
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_labeled.values,
        x=corr_labeled.columns.tolist(),
        y=corr_labeled.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr_labeled.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorbar=dict(title="r"),
    ))
    fig_corr.update_layout(
        xaxis=dict(tickangle=-45),
    )
    apply_common_layout(fig_corr, title=f"{city_for_corr}: Feature Correlation Matrix", height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

# ── 9.5 All-cities correlation comparison ───────────────────────────────────
st.header("9.5  Comparing Correlations Across Cities")

st.markdown(
    "Here's where it gets interesting: pick two features and see whether their "
    "correlation is the same across all cities. Sometimes it is. Sometimes it's not. "
    "When it's not, that tells you something about the different climatic regimes "
    "these cities live in."
)

col_f1, col_f2 = st.columns(2)
with col_f1:
    feat1 = st.selectbox("Feature A", FEATURE_COLS, index=0,
                          format_func=lambda c: FEATURE_LABELS.get(c, c), key="corr_f1")
with col_f2:
    feat2 = st.selectbox("Feature B", FEATURE_COLS, index=1,
                          format_func=lambda c: FEATURE_LABELS.get(c, c), key="corr_f2")

corr_rows = []
for city in sorted(fdf["city"].unique()):
    cd = fdf[fdf["city"] == city][[feat1, feat2]].dropna()
    if len(cd) > 2:
        r = cd[feat1].corr(cd[feat2])
        corr_rows.append({"City": city, "r": round(r, 4), "n": len(cd)})

if corr_rows:
    corr_df = pd.DataFrame(corr_rows)
    fig_bar = px.bar(
        corr_df, x="City", y="r",
        color="City", color_discrete_map=CITY_COLORS,
        title=f"Pearson r: {FEATURE_LABELS[feat1]} vs {FEATURE_LABELS[feat2]}",
        labels={"r": "Pearson r"},
    )
    fig_bar.add_hline(y=0, line_dash="dash", line_color="grey")
    apply_common_layout(fig_bar, height=400)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.dataframe(corr_df, use_container_width=True, hide_index=True)

# ── 9.6 Calendar heatmap ────────────────────────────────────────────────────
st.header("9.6  Calendar Heatmap: Daily Average Temperature")

st.markdown(
    "A calendar heatmap arranges daily values by week and day-of-week, mimicking the "
    "layout of a wall calendar. It's wonderful for spotting anomalies -- that one weirdly "
    "warm week in February, or the cold snap that hit mid-October."
)

cal_data = city_data.groupby("date")["temperature_c"].mean().reset_index()
cal_data["date"] = pd.to_datetime(cal_data["date"])
cal_data["week"] = cal_data["date"].dt.isocalendar().week.astype(int)
cal_data["year"] = cal_data["date"].dt.year
cal_data["dow"] = cal_data["date"].dt.dayofweek  # 0=Mon

# Use most recent year
latest_year = cal_data["year"].max()
cal_year = cal_data[cal_data["year"] == latest_year]

if len(cal_year) > 0:
    pivot_cal = cal_year.pivot_table(index="dow", columns="week", values="temperature_c", aggfunc="mean")
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig_cal = go.Figure(data=go.Heatmap(
        z=pivot_cal.values,
        x=[str(w) for w in pivot_cal.columns],
        y=[dow_labels[i] for i in pivot_cal.index],
        colorscale="RdYlBu_r",
        colorbar=dict(title="°C"),
    ))
    fig_cal.update_layout(
        xaxis_title="Week of Year",
        yaxis_title="Day of Week",
    )
    apply_common_layout(fig_cal, title=f"{city_sel} ({latest_year}): Daily Temperature Calendar", height=350)
    st.plotly_chart(fig_cal, use_container_width=True)

# ── Code example ─────────────────────────────────────────────────────────────
code_example(
    """import plotly.graph_objects as go
import pandas as pd

# Pivot table for heatmap
pivot = df.pivot_table(index="hour", columns="month",
                        values="temperature_c", aggfunc="mean")

# Plotly heatmap
fig = go.Figure(data=go.Heatmap(
    z=pivot.values, x=pivot.columns, y=pivot.index,
    colorscale="RdYlBu_r"
))
fig.show()

# Correlation matrix
corr = df[["temperature_c", "relative_humidity_pct",
           "wind_speed_kmh", "surface_pressure_hpa"]].corr()
fig = go.Figure(data=go.Heatmap(
    z=corr.values, x=corr.columns, y=corr.index,
    colorscale="RdBu_r", zmin=-1, zmax=1,
    text=corr.round(2).values, texttemplate="%{text}"
))
fig.show()
"""
)

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "In a correlation matrix, the diagonal values are always:",
    ["0", "0.5", "1", "Varies"],
    correct_idx=2,
    explanation="Every variable has a perfect correlation with itself (r = 1). This is trivially "
                "true and not very informative -- the interesting stuff is always in the off-diagonal "
                "cells.",
    key="ch9_quiz1",
)

quiz(
    "What does a dark red cell in a correlation heatmap (using an RdBu_r color scale, -1 to +1) indicate?",
    [
        "Strong negative correlation",
        "No correlation",
        "Strong positive correlation",
        "Missing data",
    ],
    correct_idx=2,
    explanation="On the RdBu_r (Red-Blue reversed) scale, dark red means values near +1, which "
                "is a strong positive correlation. Dark blue would mean strong negative. The white "
                "middle is near zero. Getting comfortable with color scales is one of those skills "
                "that makes heatmaps actually useful instead of just pretty.",
    key="ch9_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "Heatmaps turn numeric matrices into color patterns. Your visual system can spot patterns in color that it would miss in tables of numbers.",
    "Hour x Month temperature heatmaps are a gem: they show diurnal and seasonal cycles simultaneously in one compact image.",
    "Correlation matrices summarize all pairwise linear relationships. Focus on the off-diagonal cells -- the diagonal just tells you 1=1.",
    "Pearson captures linear relationships; Spearman captures monotonic ones. If you suspect nonlinear relationships, try Spearman.",
    "Always put the actual numbers on your heatmap cells. Color alone is not precise enough for quantitative comparisons.",
    "Calendar heatmaps are great for spotting day-level anomalies that get averaged away in other visualizations.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 8: Box Plots & Comparisons",
    prev_page="08_Box_Plots_and_Comparisons.py",
    next_label="Ch 10: Advanced Visualization",
    next_page="10_Advanced_Visualization.py",
)
