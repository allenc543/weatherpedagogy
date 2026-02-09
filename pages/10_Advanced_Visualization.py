"""Chapter 10: Advanced Visualization -- Pair plots, parallel coordinates, radar, small multiples."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, color_map, multi_subplot
from utils.constants import (
    CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS, FEATURE_UNITS,
    SEASON_ORDER,
)
from utils.stats_helpers import descriptive_stats
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(10, "Advanced Visualization", part="II")
st.markdown(
    "This chapter covers visualization techniques beyond the basics: pair plots "
    "for exploring all pairwise relationships at once, parallel coordinates for "
    "high-dimensional comparison, radar charts for city profiles, and small "
    "multiples (faceted plots) for systematic comparison."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 10.1 Pair Plot / Scatter Matrix ─────────────────────────────────────────
st.header("10.1  Pair Plot (Scatter Matrix)")

concept_box(
    "Pair Plot",
    "A pair plot (or scatter-plot matrix) shows every pairwise combination of "
    "variables in a grid. The diagonal typically shows histograms or KDEs for "
    "each variable. This lets you quickly scan for correlations, clusters, and outliers."
)

st.markdown(
    "Because the full dataset is large, we subsample for the scatter matrix."
)

pair_cities = st.multiselect(
    "Cities for pair plot",
    sorted(fdf["city"].unique()),
    default=["Dallas", "Los Angeles"],
    key="adv_pair_cities",
)
pair_sample_n = st.slider("Subsample size", 500, 5000, 2000, step=500, key="adv_pair_n")

pair_data = fdf[fdf["city"].isin(pair_cities)][FEATURE_COLS + ["city"]].dropna()
if len(pair_data) > pair_sample_n:
    pair_data = pair_data.sample(pair_sample_n, random_state=42)

if len(pair_data) > 0 and len(pair_cities) > 0:
    fig_pair = px.scatter_matrix(
        pair_data,
        dimensions=FEATURE_COLS,
        color="city",
        color_discrete_map=CITY_COLORS,
        labels=FEATURE_LABELS,
        title="Scatter Matrix of Weather Features",
        opacity=0.4,
    )
    fig_pair.update_traces(diagonal_visible=True, marker=dict(size=2))
    fig_pair.update_layout(height=700, width=800)
    apply_common_layout(fig_pair, height=700)
    st.plotly_chart(fig_pair, use_container_width=True)

    insight_box(
        "The scatter matrix shows that temperature and humidity are negatively "
        "correlated, while other pairs show weaker or no linear relationships. "
        "Color separation reveals how cities occupy different regions of feature space."
    )
else:
    st.info("Select at least one city.")

# ── 10.2 Parallel Coordinates ───────────────────────────────────────────────
st.header("10.2  Parallel Coordinates")

concept_box(
    "Parallel Coordinates",
    "In a parallel coordinates plot, each variable is a vertical axis, and each "
    "observation is a line crossing all axes. Patterns emerge when lines from the "
    "same group (e.g., city) cluster together. It can display more dimensions than "
    "a 2D scatter plot."
)

pc_sample_n = st.slider("Subsample for parallel coords", 500, 5000, 1500, step=500, key="adv_pc_n")

pc_data = fdf[FEATURE_COLS + ["city"]].dropna()
if len(pc_data) > pc_sample_n:
    pc_data = pc_data.sample(pc_sample_n, random_state=42)

if len(pc_data) > 0:
    # Map cities to numeric for color
    city_list_sorted = sorted(pc_data["city"].unique())
    city_to_num = {c: i for i, c in enumerate(city_list_sorted)}
    pc_data["city_num"] = pc_data["city"].map(city_to_num)

    dims = []
    for col in FEATURE_COLS:
        dims.append(dict(
            range=[pc_data[col].min(), pc_data[col].max()],
            label=FEATURE_LABELS.get(col, col),
            values=pc_data[col],
        ))

    fig_pc = go.Figure(data=go.Parcoords(
        line=dict(
            color=pc_data["city_num"],
            colorscale="Turbo",
            showscale=True,
            cmin=0,
            cmax=len(city_list_sorted) - 1,
            colorbar=dict(
                title="City",
                tickvals=list(range(len(city_list_sorted))),
                ticktext=city_list_sorted,
            ),
        ),
        dimensions=dims,
    ))
    apply_common_layout(fig_pc, title="Parallel Coordinates: Weather Features by City", height=500)
    st.plotly_chart(fig_pc, use_container_width=True)

    insight_box(
        "Drag along any axis to filter the data interactively. Lines that cluster "
        "tightly on one axis show low variability; spread-out lines indicate high variability."
    )

# ── 10.3 Radar Chart: City Weather Profiles ─────────────────────────────────
st.header("10.3  Radar Chart: City Weather Profiles")

concept_box(
    "Radar (Spider) Chart",
    "A radar chart plots multiple variables on axes radiating from a center. "
    "Each city becomes a polygon, making it easy to compare profiles. "
    "Variables should be normalized to the same scale."
)

# Compute per-city means and normalize to 0-1
city_means = fdf.groupby("city")[FEATURE_COLS].mean()
# Min-max normalize per feature
city_norm = (city_means - city_means.min()) / (city_means.max() - city_means.min())
city_norm = city_norm.rename(columns=FEATURE_LABELS)

radar_cities = st.multiselect(
    "Cities for radar chart",
    sorted(fdf["city"].unique()),
    default=sorted(fdf["city"].unique())[:4],
    key="adv_radar_cities",
)

if len(radar_cities) > 0:
    categories = city_norm.columns.tolist()

    fig_radar = go.Figure()
    for city in radar_cities:
        if city in city_norm.index:
            values = city_norm.loc[city].tolist()
            values.append(values[0])  # close the polygon
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=city,
                line_color=CITY_COLORS.get(city, None),
                opacity=0.6,
            ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    )
    apply_common_layout(fig_radar, title="Normalized City Weather Profiles", height=550)
    st.plotly_chart(fig_radar, use_container_width=True)

    st.caption(
        "Values are min-max normalized across cities (0 = lowest city mean, 1 = highest). "
        "This allows comparison of features with different units."
    )

    # Also show raw means table
    with st.expander("Raw city means (not normalized)"):
        raw_means = city_means.rename(columns=FEATURE_LABELS).round(2)
        st.dataframe(raw_means.loc[radar_cities], use_container_width=True)

# ── 10.4 Small Multiples: Faceted Time Series ───────────────────────────────
st.header("10.4  Small Multiples (Faceted Plots)")

concept_box(
    "Small Multiples",
    "Small multiples display the same chart type for each subset of data "
    "(e.g., one panel per city or per season). They enable direct comparison "
    "without the clutter of overlapping traces."
)

facet_feat = st.selectbox(
    "Feature to facet",
    FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    key="adv_facet_feat",
)

# Monthly averages per city
monthly = (
    fdf.groupby(["city", "month"])[facet_feat]
    .mean()
    .reset_index()
)
monthly["month_name"] = monthly["month"].apply(lambda m: pd.Timestamp(2024, m, 1).strftime("%b"))
# Enforce month ordering
month_order = [pd.Timestamp(2024, m, 1).strftime("%b") for m in range(1, 13)]

if len(monthly) > 0:
    fig_facet = px.line(
        monthly, x="month_name", y=facet_feat, color="city",
        facet_col="city", facet_col_wrap=3,
        color_discrete_map=CITY_COLORS,
        labels={facet_feat: FEATURE_LABELS.get(facet_feat, facet_feat),
                "month_name": "Month"},
        category_orders={"month_name": month_order},
        title=f"Monthly Average {FEATURE_LABELS[facet_feat]} by City",
        markers=True,
    )
    fig_facet.update_layout(showlegend=False, height=600)
    fig_facet.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    apply_common_layout(fig_facet, height=600)
    st.plotly_chart(fig_facet, use_container_width=True)

# ── 10.5 Faceted Histograms by Season ───────────────────────────────────────
st.header("10.5  Faceted Histograms by Season")

facet_city = st.selectbox(
    "City for seasonal facets", sorted(fdf["city"].unique()),
    key="adv_facet_hist_city",
)

season_hist = fdf[fdf["city"] == facet_city][[facet_feat, "season"]].dropna()
season_hist["season"] = pd.Categorical(season_hist["season"], categories=SEASON_ORDER, ordered=True)

if len(season_hist) > 0:
    fig_shist = px.histogram(
        season_hist, x=facet_feat, facet_col="season",
        color="season", nbins=40,
        category_orders={"season": SEASON_ORDER},
        labels={facet_feat: FEATURE_LABELS.get(facet_feat, facet_feat)},
        title=f"{facet_city}: {FEATURE_LABELS[facet_feat]} Distribution by Season",
    )
    fig_shist.update_layout(showlegend=False, height=400)
    fig_shist.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    apply_common_layout(fig_shist, height=400)
    st.plotly_chart(fig_shist, use_container_width=True)

# ── 10.6 Sunburst: hierarchical view ────────────────────────────────────────
st.header("10.6  Sunburst Chart: Hierarchical Grouping")

concept_box(
    "Sunburst Chart",
    "Sunburst charts show hierarchical data as concentric rings. Each ring is a "
    "level in the hierarchy (e.g., city then season). The size/color of each segment "
    "can encode a numeric variable."
)

sun_data = (
    fdf.groupby(["city", "season"])["temperature_c"]
    .mean()
    .reset_index()
)
sun_data["season"] = pd.Categorical(sun_data["season"], categories=SEASON_ORDER, ordered=True)

if len(sun_data) > 0:
    fig_sun = px.sunburst(
        sun_data, path=["city", "season"], values=None,
        color="temperature_c", color_continuous_scale="RdYlBu_r",
        title="City > Season Hierarchy (color = avg temperature)",
    )
    apply_common_layout(fig_sun, height=550)
    st.plotly_chart(fig_sun, use_container_width=True)

# ── 10.7 Grouped bar chart ──────────────────────────────────────────────────
st.header("10.7  Grouped Bar Chart: Seasonal Averages")

bar_data = (
    fdf.groupby(["city", "season"])[facet_feat]
    .mean()
    .reset_index()
)
bar_data["season"] = pd.Categorical(bar_data["season"], categories=SEASON_ORDER, ordered=True)
bar_data = bar_data.sort_values("season")

if len(bar_data) > 0:
    fig_grouped = px.bar(
        bar_data, x="season", y=facet_feat, color="city",
        barmode="group",
        color_discrete_map=CITY_COLORS,
        labels={facet_feat: FEATURE_LABELS.get(facet_feat, facet_feat), "season": "Season"},
        title=f"Seasonal Average {FEATURE_LABELS[facet_feat]} by City",
        category_orders={"season": SEASON_ORDER},
    )
    apply_common_layout(fig_grouped, height=500)
    st.plotly_chart(fig_grouped, use_container_width=True)

# ── Code example ─────────────────────────────────────────────────────────────
code_example(
    """import plotly.express as px
import plotly.graph_objects as go

# Scatter matrix (pair plot)
fig = px.scatter_matrix(df, dimensions=feature_cols, color="city",
                         opacity=0.3)
fig.update_traces(diagonal_visible=True, marker=dict(size=2))
fig.show()

# Parallel coordinates
fig = go.Figure(data=go.Parcoords(
    line=dict(color=city_numeric, colorscale="Turbo"),
    dimensions=[
        dict(label="Temp", values=df["temperature_c"]),
        dict(label="Humidity", values=df["relative_humidity_pct"]),
        dict(label="Wind", values=df["wind_speed_kmh"]),
        dict(label="Pressure", values=df["surface_pressure_hpa"]),
    ]
))
fig.show()

# Radar chart
fig = go.Figure()
for city in cities:
    fig.add_trace(go.Scatterpolar(
        r=normalized_values[city], theta=categories,
        fill="toself", name=city))
fig.show()

# Faceted (small multiples)
fig = px.line(monthly, x="month", y="temperature_c",
              facet_col="city", facet_col_wrap=3)
fig.show()
"""
)

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "What is the main advantage of a pair plot (scatter matrix)?",
    [
        "It shows time series trends",
        "It displays all pairwise variable relationships simultaneously",
        "It reduces dimensionality",
        "It performs statistical tests",
    ],
    correct_idx=1,
    explanation="A pair plot creates a grid of scatter plots for every pair of variables, "
                "allowing quick visual scanning for correlations and patterns.",
    key="ch10_quiz1",
)

quiz(
    "Why should you normalize variables before creating a radar chart?",
    [
        "To make the chart render faster",
        "To ensure all axes use the same scale for fair comparison",
        "To remove outliers",
        "Normalization is not needed for radar charts",
    ],
    correct_idx=1,
    explanation="Without normalization, a variable with a large range (e.g., pressure in hPa) "
                "would dominate the chart and others would be invisible.",
    key="ch10_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "Pair plots show all pairwise scatter plots in one view -- great for exploratory analysis.",
    "Parallel coordinates display high-dimensional data; each axis is a variable.",
    "Radar charts compare profiles (e.g., city weather profiles) on a common normalized scale.",
    "Small multiples (faceted plots) enable clean comparison without overlapping traces.",
    "Always subsample large datasets for pair plots and parallel coordinates to keep charts responsive.",
    "Choose the chart type that best matches your question: relationships (scatter), comparison (bar/radar), distribution (histogram/violin), or composition (sunburst/treemap).",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 9: Heatmaps & Correlation",
    prev_page="09_Heatmaps_and_Correlation.py",
)
