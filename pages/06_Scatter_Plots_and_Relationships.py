"""Chapter 6: Scatter Plots & Relationships -- Bivariate plots, overplotting, trend lines."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import scatter_chart, apply_common_layout, color_map
from utils.constants import (
    CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS, FEATURE_UNITS,
)
from utils.stats_helpers import correlation_matrix
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(6, "Scatter Plots & Relationships", part="II")
st.markdown(
    "So far we've been looking at one variable at a time. But the really interesting "
    "questions are about *relationships*: does high temperature come with low humidity? "
    "Does wind speed affect pressure? The scatter plot is the tool for this, and it's "
    "the most honest chart type in existence -- every dot is a real observation, and "
    "any pattern you see (or don't see) is right there in the data. There's also a "
    "**practical problem** you'll hit immediately with 100,000+ points, and we'll solve it."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 6.1 Theory ──────────────────────────────────────────────────────────────
st.header("6.1  What Does a Scatter Plot Show?")

concept_box(
    "Reading Bivariate Relationships",
    "Each dot is one observation, with its X and Y position determined by two variables. "
    "When you stare at a scatter plot, look for three things:<br>"
    "- <b>Direction</b>: do the dots trend up-right (positive) or down-right (negative)?<br>"
    "- <b>Form</b>: is the relationship linear, curved, clustered, or just... nothing?<br>"
    "- <b>Strength</b>: how tightly do points follow the trend? A tight band means strong "
    "correlation; a diffuse cloud means weak or no correlation."
)

formula_box(
    "Pearson Correlation Coefficient",
    r"\underbrace{r}_{\text{Pearson correlation}} = \frac{\sum (\underbrace{x_i}_{\text{x observation}} - \underbrace{\bar{x}}_{\text{x mean}})(\underbrace{y_i}_{\text{y observation}} - \underbrace{\bar{y}}_{\text{y mean}})}"
    r"{\sqrt{\sum(x_i-\bar{x})^2 \sum(y_i-\bar{y})^2}}",
    "r ranges from -1 (perfect negative linear relationship) to +1 (perfect positive). "
    "r = 0 means no *linear* relationship -- but be careful, there could still be a strong "
    "nonlinear one. r only sees straight lines.",
)

# ── 6.2 Interactive scatter ─────────────────────────────────────────────────
st.header("6.2  Interactive Scatter Plot")

col_x, col_y = st.columns(2)
with col_x:
    x_feat = st.selectbox(
        "X-axis feature", FEATURE_COLS,
        index=0,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
        key="scatter_x",
    )
with col_y:
    y_feat = st.selectbox(
        "Y-axis feature", FEATURE_COLS,
        index=1,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
        key="scatter_y",
    )

cities_sel = st.multiselect(
    "Cities to include",
    sorted(fdf["city"].unique()),
    default=sorted(fdf["city"].unique()),
    key="scatter_cities",
)

show_trendline = st.checkbox("Show OLS trend line per city", value=False, key="scatter_trend")
point_opacity = st.slider("Point opacity", 0.05, 1.0, 0.2, step=0.05, key="scatter_opacity")

subset = fdf[fdf["city"].isin(cities_sel)][[x_feat, y_feat, "city"]].dropna()

if len(subset) > 0:
    trendline = "ols" if show_trendline else None
    fig = px.scatter(
        subset, x=x_feat, y=y_feat, color="city",
        color_discrete_map=CITY_COLORS,
        opacity=point_opacity,
        trendline=trendline,
        labels={**FEATURE_LABELS, "city": "City"},
        title=f"{FEATURE_LABELS[x_feat]} vs {FEATURE_LABELS[y_feat]}",
    )
    apply_common_layout(fig, height=550)
    st.plotly_chart(fig, use_container_width=True)

    # Compute correlation per city
    st.subheader("Per-City Correlation (Pearson r)")
    corr_rows = []
    for city in cities_sel:
        cs = subset[subset["city"] == city]
        if len(cs) > 2:
            r = cs[x_feat].corr(cs[y_feat])
            corr_rows.append({"City": city, "r": round(r, 4), "n": len(cs)})
    if corr_rows:
        st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)

# ── 6.3 Temperature vs Humidity: a classic negative correlation ──────────────
st.header("6.3  Case Study: Temperature vs Humidity")

insight_box(
    "Here's a relationship that seems backwards until you think about it: as temperature "
    "goes *up*, relative humidity often goes *down*. Why? Because warmer air can hold more "
    "moisture, so the same amount of water vapor fills a smaller *fraction* of the air's "
    "capacity. The absolute moisture might not change at all, but the *relative* humidity "
    "drops. This is a genuinely useful thing to understand if you've ever wondered why "
    "humid summers feel even worse than the humidity numbers suggest."
)

th_data = fdf[["temperature_c", "relative_humidity_pct", "city"]].dropna()
if len(th_data) > 0:
    fig_th = px.scatter(
        th_data, x="temperature_c", y="relative_humidity_pct", color="city",
        color_discrete_map=CITY_COLORS, opacity=0.15,
        labels=FEATURE_LABELS,
        title="Temperature vs Relative Humidity",
    )
    apply_common_layout(fig_th, height=500)
    st.plotly_chart(fig_th, use_container_width=True)

    overall_r = th_data["temperature_c"].corr(th_data["relative_humidity_pct"])
    st.write(f"**Overall Pearson r = {overall_r:.3f}** -- a clear negative correlation.")

# ── 6.4 Overplotting solutions ───────────────────────────────────────────────
st.header("6.4  Dealing with Overplotting")

concept_box(
    "The 100,000-Point Problem",
    "With enough data points, your scatter plot turns into a solid blob. All structure "
    "disappears. This isn't a flaw in the data -- it's a flaw in naive scatter plots. "
    "Solutions:<br>"
    "- <b>Transparency</b> (low opacity) so dense regions appear darker.<br>"
    "- <b>Hexbin / 2D histograms</b> that count points per cell and use color for density.<br>"
    "- <b>Subsampling</b> -- just plot fewer points (sacrilege to purists, pragmatic for everyone else).<br>"
    "- <b>Contour / density plots</b> that estimate density as smooth curves."
)

st.markdown(
    "Let's compare three approaches for the same temperature-vs-humidity data. "
    "Same underlying data, very different visual stories."
)

city_overplot = st.selectbox(
    "City for overplotting demo", sorted(fdf["city"].unique()),
    key="scatter_overplot_city",
)
op_data = fdf.loc[fdf["city"] == city_overplot, ["temperature_c", "relative_humidity_pct"]].dropna()

if len(op_data) > 0:
    tab1, tab2, tab3 = st.tabs(["Transparency", "2D Histogram (Hexbin)", "Contour Density"])

    with tab1:
        fig_t = px.scatter(
            op_data, x="temperature_c", y="relative_humidity_pct",
            opacity=0.1, labels=FEATURE_LABELS,
            title=f"{city_overplot}: Scatter with alpha = 0.1",
        )
        fig_t.update_traces(marker=dict(size=3, color=CITY_COLORS.get(city_overplot, "blue")))
        apply_common_layout(fig_t, height=450)
        st.plotly_chart(fig_t, use_container_width=True)

    with tab2:
        fig_hex = px.density_heatmap(
            op_data, x="temperature_c", y="relative_humidity_pct",
            nbinsx=50, nbinsy=50, color_continuous_scale="Blues",
            labels=FEATURE_LABELS,
            title=f"{city_overplot}: 2D Histogram",
        )
        apply_common_layout(fig_hex, height=450)
        st.plotly_chart(fig_hex, use_container_width=True)

    with tab3:
        fig_contour = px.density_contour(
            op_data, x="temperature_c", y="relative_humidity_pct",
            labels=FEATURE_LABELS,
            title=f"{city_overplot}: Density Contour",
        )
        fig_contour.update_traces(contours_coloring="fill", colorscale="Blues")
        apply_common_layout(fig_contour, height=450)
        st.plotly_chart(fig_contour, use_container_width=True)

# ── 6.5 Subsample comparison ────────────────────────────────────────────────
st.header("6.5  Subsampling for Clarity")

subsample_n = st.slider(
    "Points to display", 500, 10000, 3000, step=500,
    key="scatter_subsample",
)

all_data = fdf[FEATURE_COLS + ["city"]].dropna()
if len(all_data) > subsample_n:
    sample = all_data.sample(subsample_n, random_state=42)
else:
    sample = all_data

fig_sub = px.scatter(
    sample, x=x_feat, y=y_feat, color="city",
    color_discrete_map=CITY_COLORS, opacity=0.5,
    labels=FEATURE_LABELS,
    title=f"Subsampled Scatter ({len(sample):,} points)",
)
apply_common_layout(fig_sub, height=500)
st.plotly_chart(fig_sub, use_container_width=True)

# ── Code example ─────────────────────────────────────────────────────────────
code_example(
    """import plotly.express as px

# Basic scatter with color by city
fig = px.scatter(df, x="temperature_c", y="relative_humidity_pct",
                 color="city", opacity=0.2, trendline="ols")
fig.show()

# 2D histogram for overplotting
fig = px.density_heatmap(df, x="temperature_c", y="relative_humidity_pct",
                          nbinsx=50, nbinsy=50)
fig.show()

# Pearson correlation
r = df["temperature_c"].corr(df["relative_humidity_pct"])
print(f"Pearson r = {r:.3f}")
"""
)

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "A Pearson r of -0.6 indicates:",
    [
        "No relationship",
        "A strong positive relationship",
        "A moderate negative linear relationship",
        "A perfect negative relationship",
    ],
    correct_idx=2,
    explanation="r = -0.6 is a moderately strong negative linear association. It's not nothing "
                "(that would be r near 0) and it's not perfect (that would be r = -1). It means: "
                "when one variable goes up, the other tends to go down, but with plenty of scatter.",
    key="ch6_quiz1",
)

quiz(
    "Which technique is BEST for showing density in a scatter plot with 100,000 points?",
    [
        "Increasing point size",
        "Using a 2D histogram or hexbin",
        "Adding jitter",
        "Changing the axis scale",
    ],
    correct_idx=1,
    explanation="A 2D histogram or hexbin counts points per cell and uses color intensity to show "
                "density. It's the right tool because it turns the overplotting problem into useful "
                "information -- dense regions pop out, sparse regions fade.",
    key="ch6_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "Scatter plots reveal the direction, form, and strength of bivariate relationships. They're the most honest chart type we have.",
    "Pearson r quantifies linear correlation (-1 to +1). But remember: r = 0 doesn't mean 'no relationship,' it means 'no *linear* relationship.'",
    "Temperature and relative humidity show a clear negative correlation -- warmer air has more capacity, so the same moisture reads as lower relative humidity.",
    "With large datasets, naive scatter plots become useless blobs. Use transparency, 2D histograms, or contour plots to reveal structure.",
    "Always check per-city correlations separately. Aggregating across groups can mask or inflate patterns (Simpson's paradox is real and it will get you).",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 5: Histograms & Density",
    prev_page="05_Histograms_and_Density.py",
    next_label="Ch 7: Time Series Visualization",
    next_page="07_Time_Series_Visualization.py",
)
