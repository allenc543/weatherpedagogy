"""Chapter 8: Box Plots & Comparisons -- Quartiles, outliers, violins, strip plots."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import box_chart, apply_common_layout, color_map
from utils.constants import (
    CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS, FEATURE_UNITS,
    SEASON_ORDER,
)
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(8, "Box Plots & Comparisons", part="II")
st.markdown(
    "Box plots are the statistical equivalent of a headshot: they compress an entire "
    "distribution into a compact glyph that tells you the center, the spread, the "
    "symmetry, and the outliers -- all at a glance. They're the go-to chart for "
    "comparing distributions across groups. This chapter also introduces violin plots "
    "and strip plots, which each sacrifice something to gain something else. "
    "**Understanding these tradeoffs is what makes you a good data visualizer.**"
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 8.1 Anatomy of a box plot ───────────────────────────────────────────────
st.header("8.1  Anatomy of a Box Plot")

concept_box(
    "How to Read a Box Plot (It's More Informative Than It Looks)",
    "- <b>The box</b>: spans from Q1 (25th percentile) to Q3 (75th percentile). "
    "This is the IQR -- the middle 50% of your data lives here.<br>"
    "- <b>The line inside the box</b>: that's the median. If it's not centered in "
    "the box, the distribution is skewed.<br>"
    "- <b>The whiskers</b>: extend to the farthest data point within 1.5 * IQR of "
    "the box edges. Why 1.5? It's a convention that John Tukey proposed, and it works "
    "well in practice -- about 99.3% of normally-distributed data falls within the whiskers.<br>"
    "- <b>The dots beyond the whiskers</b>: potential outliers. 'Potential' because a dot "
    "beyond the fence isn't necessarily an error -- it might be a genuine extreme value."
)

formula_box(
    "Outlier Fences",
    r"\text{Lower fence} = Q1 - 1.5 \times IQR, \quad \text{Upper fence} = Q3 + 1.5 \times IQR",
    "Anything outside these fences gets plotted as an individual point. The 1.5 multiplier "
    "is Tukey's convention -- not derived from first principles, but empirically excellent.",
)

# ── 8.2 Interactive: feature by city ─────────────────────────────────────────
st.header("8.2  Interactive: Feature by City")

feat_sel = st.selectbox(
    "Feature",
    FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    key="box_feat",
)

fig_city = box_chart(
    fdf, x="city", y=feat_sel, color="city",
    title=f"{FEATURE_LABELS[feat_sel]} by City",
)
st.plotly_chart(fig_city, use_container_width=True)

insight_box(
    "Compare the box widths. Dallas (temperature) has a box you could drive a truck through -- "
    "enormous variability. Los Angeles has a narrow little box, suggesting its weather stays "
    "within a tight range. You can read the same story from standard deviations, but the box "
    "plot makes it *visible* in a way that numbers alone don't."
)

# ── 8.3 Grouping by season ──────────────────────────────────────────────────
st.header("8.3  Temperature by Season")

city_box = st.selectbox(
    "City for seasonal box plot", sorted(fdf["city"].unique()),
    key="box_season_city",
)

season_data = fdf[fdf["city"] == city_box][["season", feat_sel]].dropna()
# Enforce season order
season_data["season"] = pd.Categorical(season_data["season"], categories=SEASON_ORDER, ordered=True)
season_data = season_data.sort_values("season")

if len(season_data) > 0:
    fig_season = px.box(
        season_data, x="season", y=feat_sel,
        color="season",
        labels={feat_sel: FEATURE_LABELS.get(feat_sel, feat_sel), "season": "Season"},
        title=f"{city_box}: {FEATURE_LABELS[feat_sel]} by Season",
    )
    apply_common_layout(fig_season, height=500)
    st.plotly_chart(fig_season, use_container_width=True)

# ── 8.4 Monthly box plot ────────────────────────────────────────────────────
st.header("8.4  Monthly Box Plot")

month_data = fdf[fdf["city"] == city_box][["month", "month_name", feat_sel]].dropna()
month_data = month_data.sort_values("month")

if len(month_data) > 0:
    fig_month = px.box(
        month_data, x="month_name", y=feat_sel,
        labels={feat_sel: FEATURE_LABELS.get(feat_sel, feat_sel), "month_name": "Month"},
        title=f"{city_box}: {FEATURE_LABELS[feat_sel]} by Month",
        category_orders={"month_name": month_data.drop_duplicates("month").sort_values("month")["month_name"].tolist()},
    )
    apply_common_layout(fig_month, height=500)
    st.plotly_chart(fig_month, use_container_width=True)

# ── 8.5 Violin plots ────────────────────────────────────────────────────────
st.header("8.5  Violin Plots")

concept_box(
    "Violin Plots: Box Plots With a PhD",
    "You might be thinking: box plots are great, but they throw away all the detail "
    "about distribution shape. What if the data is bimodal -- two humps? A box plot "
    "would hide that completely. Enter the violin plot: it takes a box plot and adds "
    "a mirrored KDE on each side, showing you the full distribution shape. It's more "
    "ink for more information -- the right tradeoff when shape matters."
)

show_box_inside = st.checkbox("Show box plot inside violin", value=True, key="violin_box")
box_param = "all" if show_box_inside else False

fig_violin = px.violin(
    fdf, x="city", y=feat_sel, color="city",
    color_discrete_map=CITY_COLORS,
    box=show_box_inside,
    labels={feat_sel: FEATURE_LABELS.get(feat_sel, feat_sel), "city": "City"},
    title=f"Violin Plot: {FEATURE_LABELS[feat_sel]} by City",
)
apply_common_layout(fig_violin, height=550)
st.plotly_chart(fig_violin, use_container_width=True)

# ── 8.6 Strip (jitter) plots ────────────────────────────────────────────────
st.header("8.6  Strip Plots")

concept_box(
    "Strip Plots: Every Point Gets Its Day",
    "A strip plot is the maximally honest chart: it shows *every single data point*, "
    "with random horizontal jitter to avoid overlap. The upside: nothing is hidden or "
    "summarized. The downside: with more than a few thousand points, it becomes an "
    "unreadable mess. Best for smaller datasets or subsets where you genuinely want to "
    "see individual observations."
)

strip_city = st.selectbox(
    "City for strip plot", sorted(fdf["city"].unique()),
    key="strip_city",
)

strip_data = fdf[fdf["city"] == strip_city][["season", feat_sel]].dropna()
strip_data["season"] = pd.Categorical(strip_data["season"], categories=SEASON_ORDER, ordered=True)

# Subsample for readability
max_strip = 2000
if len(strip_data) > max_strip:
    strip_data = strip_data.sample(max_strip, random_state=42)

if len(strip_data) > 0:
    fig_strip = px.strip(
        strip_data, x="season", y=feat_sel,
        color="season",
        labels={feat_sel: FEATURE_LABELS.get(feat_sel, feat_sel), "season": "Season"},
        title=f"{strip_city}: Strip Plot of {FEATURE_LABELS[feat_sel]} by Season",
    )
    fig_strip.update_traces(jitter=0.4, marker=dict(size=3, opacity=0.4))
    apply_common_layout(fig_strip, height=500)
    st.plotly_chart(fig_strip, use_container_width=True)

# ── 8.7 Side-by-side: box vs violin vs strip ────────────────────────────────
st.header("8.7  Comparison: Box vs Violin vs Strip")

st.markdown(
    "Same data, three different chart types. Each one shows you something the others "
    "hide. The box plot gives you quick summary stats. The violin reveals distribution "
    "shape. The strip shows individual points. Which one is 'best'? Depends entirely "
    "on your question."
)

compare_data = fdf[["city", feat_sel]].dropna()

tab1, tab2, tab3 = st.tabs(["Box", "Violin", "Strip"])
with tab1:
    fig_b = px.box(compare_data, x="city", y=feat_sel, color="city",
                   color_discrete_map=CITY_COLORS, labels=FEATURE_LABELS)
    apply_common_layout(fig_b, title="Box Plot", height=450)
    st.plotly_chart(fig_b, use_container_width=True)

with tab2:
    fig_v = px.violin(compare_data, x="city", y=feat_sel, color="city",
                      color_discrete_map=CITY_COLORS, box=True, labels=FEATURE_LABELS)
    apply_common_layout(fig_v, title="Violin Plot", height=450)
    st.plotly_chart(fig_v, use_container_width=True)

with tab3:
    strip_sample = compare_data.sample(min(3000, len(compare_data)), random_state=42)
    fig_s = px.strip(strip_sample, x="city", y=feat_sel, color="city",
                     color_discrete_map=CITY_COLORS, labels=FEATURE_LABELS)
    fig_s.update_traces(jitter=0.4, marker=dict(size=2, opacity=0.3))
    apply_common_layout(fig_s, title="Strip Plot (sampled)", height=450)
    st.plotly_chart(fig_s, use_container_width=True)

# ── Code example ─────────────────────────────────────────────────────────────
code_example(
    """import plotly.express as px

# Box plot by city
fig = px.box(df, x="city", y="temperature_c", color="city",
             title="Temperature by City")
fig.show()

# Violin plot
fig = px.violin(df, x="city", y="temperature_c", color="city",
                box=True, title="Temperature Violin Plot")
fig.show()

# Box plot by month
fig = px.box(df, x="month_name", y="temperature_c",
             category_orders={"month_name": month_order})
fig.show()

# Outlier detection using IQR
Q1 = df["wind_speed_kmh"].quantile(0.25)
Q3 = df["wind_speed_kmh"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["wind_speed_kmh"] < Q1 - 1.5*IQR) |
              (df["wind_speed_kmh"] > Q3 + 1.5*IQR)]
print(f"Outliers: {len(outliers)} rows")
"""
)

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "In a standard box plot, the whiskers extend to:",
    [
        "The min and max of the data",
        "One standard deviation from the mean",
        "The farthest data point within 1.5 * IQR from the box",
        "The 5th and 95th percentiles",
    ],
    correct_idx=2,
    explanation="The whiskers reach out to the most extreme data point that's still within "
                "1.5 * IQR of Q1 or Q3. Anything beyond that gets plotted as an individual dot. "
                "This is Tukey's convention, and it's one of those 'because it works well' rules "
                "rather than a derivation from first principles.",
    key="ch8_quiz1",
)

quiz(
    "What advantage does a violin plot have over a box plot?",
    [
        "It shows the mean",
        "It reveals the full distribution shape, including multi-modality",
        "It requires less data",
        "It is faster to compute",
    ],
    correct_idx=1,
    explanation="The KDE 'violin' shape shows you things a box plot simply can't: bimodal "
                "distributions, unusual bumps, asymmetric tails. A box plot compresses all of that "
                "into five numbers, which is efficient but lossy.",
    key="ch8_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "Box plots pack five summary stats into one compact glyph: min (within fence), Q1, median, Q3, max (within fence). Hard to beat for group comparisons.",
    "Points beyond the 1.5 * IQR whiskers are flagged as potential outliers -- 'potential' because extreme doesn't always mean wrong.",
    "Violin plots add a KDE to show distribution shape, revealing multi-modality that box plots would hide entirely.",
    "Strip/jitter plots show every individual data point. Maximum honesty, but they break down with large datasets.",
    "There's no single 'best' chart -- box, violin, and strip each trade off information density against visual clarity. Pick the one that answers your question.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 7: Time Series Visualization",
    prev_page="07_Time_Series_Visualization.py",
    next_label="Ch 9: Heatmaps & Correlation",
    next_page="09_Heatmaps_and_Correlation.py",
)
