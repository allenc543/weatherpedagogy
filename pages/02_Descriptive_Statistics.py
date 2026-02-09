"""Chapter 2: Descriptive Statistics -- mean, median, std, skewness, kurtosis, IQR."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import (
    histogram_chart, box_chart, apply_common_layout, color_map,
)
from utils.constants import (
    CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS, FEATURE_UNITS,
)
from utils.stats_helpers import descriptive_stats
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(2, "Descriptive Statistics", part="I")
st.markdown(
    "You've got a dataset. Great. Now: what does it *look* like? Not in the visual "
    "sense -- we'll get to charts later -- but numerically. Can you summarize its "
    "center, its spread, its shape, all in a few numbers? That's what descriptive "
    "statistics are for, and they're **deceptively deep** once you start thinking "
    "about why we compute them the way we do."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 2.1 Measures of Central Tendency ─────────────────────────────────────────
st.header("2.1  Measures of Central Tendency")

concept_box(
    "Mean, Median, and Mode: Three Ways to Find the 'Center'",
    "<b>Mean</b> -- the arithmetic average. Add everything up, divide by the count. "
    "Simple, elegant, and exquisitely sensitive to outliers. One billionaire walks "
    "into a bar and the average net worth goes through the roof.<br>"
    "<b>Median</b> -- sort all the values, pick the one in the middle. It doesn't care "
    "about that billionaire at all. Robust to outliers, but ignores a lot of information.<br>"
    "<b>Mode</b> -- the most common value. Mostly useful for categorical data (the most "
    "popular ice cream flavor). For continuous data it's usually not very helpful."
)

formula_box("Arithmetic Mean", r"\underbrace{\bar{x}}_{\text{sample mean}} = \frac{1}{\underbrace{n}_{\text{sample size}}}\sum_{i=1}^{n} \underbrace{x_i}_{\text{each observation}}")

# ── 2.2 Measures of Spread ──────────────────────────────────────────────────
st.header("2.2  Measures of Spread")

concept_box(
    "Standard Deviation & IQR: How Spread Out Is This Thing?",
    "Knowing the center isn't enough. Two cities could have the same average temperature "
    "but feel completely different to live in -- one swings from freezing to blistering, "
    "the other stays pleasantly mild.<br><br>"
    "<b>Variance</b> (s²) measures how far values spread from the mean. But why do we "
    "*square* the deviations? Because if we just averaged the raw deviations, positive and "
    "negative would cancel out and we'd get zero. Squaring solves that -- and has nice "
    "mathematical properties, though it gives us units² which is weird.<br>"
    "<b>Standard Deviation</b> (s) is just the square root of variance, which brings us "
    "back to the original units. Thank goodness.<br>"
    "<b>IQR</b> = Q3 - Q1, the range of the middle 50%. Like the median, it ignores "
    "extremes entirely."
)

formula_box(
    "Sample Standard Deviation",
    r"\underbrace{s}_{\text{sample std dev}} = \sqrt{\frac{1}{\underbrace{n-1}_{\text{Bessel's correction}}}\sum_{i=1}^{n}(\underbrace{x_i}_{\text{observation}} - \underbrace{\bar{x}}_{\text{sample mean}})^2}",
    "Why n-1 instead of n? This is Bessel's correction. When you estimate spread "
    "from a sample, dividing by n slightly underestimates the true variance. Dividing "
    "by n-1 fixes that. It's one of those things that sounds like a technicality until "
    "you realize it matters a lot with small samples.",
)

# ── 2.3 Shape: Skewness & Kurtosis ──────────────────────────────────────────
st.header("2.3  Shape: Skewness & Kurtosis")

concept_box(
    "Skewness & Kurtosis: The Shape of the Distribution",
    "<b>Skewness</b> tells you about asymmetry. Imagine someone grabs the right tail "
    "of your bell curve and stretches it out -- that's positive skew. The left tail is "
    "longer? Negative skew. Zero means it's symmetric. Wind speed, for example, is "
    "positively skewed because you can't have negative wind but you can have a hurricane.<br>"
    "<b>Kurtosis</b> (excess) is about tail heaviness compared to a normal distribution. "
    "High kurtosis means more extreme values than you'd expect from a normal. It's not "
    "about 'peakedness' despite what some textbooks say -- it's really about the tails."
)

# ── 2.4 Interactive: per-city stats comparison ───────────────────────────────
st.header("2.4  Interactive: Per-City Statistics")

feature = st.selectbox(
    "Select a weather feature",
    FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    key="desc_feature",
)

# Compute stats for each city
rows = []
for city in sorted(fdf["city"].unique()):
    s = fdf.loc[fdf["city"] == city, feature].dropna()
    stats = descriptive_stats(s)
    stats["city"] = city
    rows.append(stats)

stats_df = pd.DataFrame(rows).set_index("city")
stats_df = stats_df[["count", "mean", "median", "std", "min", "max", "q25", "q75", "iqr", "skewness", "kurtosis"]]
stats_df = stats_df.round(3)
st.dataframe(stats_df, use_container_width=True)

# ── Bar chart of means and stds ──────────────────────────────────────────────
st.subheader("Mean with Standard Deviation Error Bars")
bar_data = stats_df.reset_index()
fig = go.Figure()
for _, row in bar_data.iterrows():
    fig.add_trace(go.Bar(
        x=[row["city"]],
        y=[row["mean"]],
        error_y=dict(type="data", array=[row["std"]]),
        name=row["city"],
        marker_color=CITY_COLORS.get(row["city"], "#636EFA"),
    ))
fig.update_layout(showlegend=False, yaxis_title=FEATURE_LABELS.get(feature, feature))
apply_common_layout(fig, title=f"Mean {FEATURE_LABELS[feature]} by City", height=400)
st.plotly_chart(fig, use_container_width=True)

# ── 2.5 Deep Dive: Dallas vs Los Angeles ────────────────────────────────────
st.header("2.5  Deep Dive: Dallas vs Los Angeles")

insight_box(
    "This is where descriptive statistics stop being abstract and start telling a story. "
    "Dallas has brutal summers and genuinely cold winters -- its temperature swings are "
    "enormous. Los Angeles, blessed by geography, sits in a narrow mild range year-round. "
    "Their standard deviations capture this difference perfectly: Dallas is spread out, "
    "LA is concentrated. Same planet, very different experiences."
)

col1, col2 = st.columns(2)

for col, city in zip([col1, col2], ["Dallas", "Los Angeles"]):
    with col:
        city_data = fdf.loc[fdf["city"] == city, feature].dropna()
        if len(city_data) == 0:
            st.info(f"No data for {city} with current filters.")
            continue
        cs = descriptive_stats(city_data)
        st.subheader(city)
        st.write(f"Mean: **{cs['mean']:.2f}** {FEATURE_UNITS.get(feature, '')}")
        st.write(f"Std:  **{cs['std']:.2f}** {FEATURE_UNITS.get(feature, '')}")
        st.write(f"IQR:  **{cs['iqr']:.2f}** {FEATURE_UNITS.get(feature, '')}")
        st.write(f"Skewness: **{cs['skewness']:.3f}**")
        st.write(f"Range: {cs['min']:.1f} to {cs['max']:.1f}")

# Side-by-side histograms
dallas_la = fdf[fdf["city"].isin(["Dallas", "Los Angeles"])]
if len(dallas_la) > 0:
    fig_cmp = histogram_chart(
        dallas_la, x=feature, title=f"Dallas vs LA: {FEATURE_LABELS[feature]}", nbins=60,
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

# ── 2.6 Pick a city ─────────────────────────────────────────────────────────
st.header("2.6  Explore Any City")

city_pick = st.selectbox("City", sorted(fdf["city"].unique()), key="desc_city_pick")
feat_pick = st.selectbox(
    "Feature",
    FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    key="desc_feat_pick",
)

city_series = fdf.loc[fdf["city"] == city_pick, feat_pick].dropna()
if len(city_series) > 0:
    cs2 = descriptive_stats(city_series)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean", f"{cs2['mean']:.2f}")
    m2.metric("Median", f"{cs2['median']:.2f}")
    m3.metric("Std Dev", f"{cs2['std']:.2f}")
    m4.metric("IQR", f"{cs2['iqr']:.2f}")

    fig_h = px.histogram(
        city_series, nbins=50,
        labels={"value": FEATURE_LABELS.get(feat_pick, feat_pick)},
        title=f"{city_pick} -- {FEATURE_LABELS[feat_pick]} Distribution",
    )
    # Add mean and median lines
    fig_h.add_vline(x=cs2["mean"], line_dash="dash", line_color="red",
                    annotation_text="Mean", annotation_position="top right")
    fig_h.add_vline(x=cs2["median"], line_dash="dot", line_color="blue",
                    annotation_text="Median", annotation_position="top left")
    apply_common_layout(fig_h, height=400)
    st.plotly_chart(fig_h, use_container_width=True)

    if abs(cs2["skewness"]) < 0.5:
        insight_box("The skewness is close to zero, which means the distribution is roughly "
                    "symmetric. Notice how the mean and median are sitting almost on top of "
                    "each other -- that's what you'd expect when neither tail is doing anything unusual.")
    elif cs2["skewness"] > 0:
        insight_box("This distribution is positively skewed -- the right tail stretches out "
                    "further than the left. What this means in practice: there are occasional "
                    "unusually high values pulling the mean above the median. The mean is being "
                    "'fooled' by extremes; the median holds steady.")
    else:
        insight_box("Negative skew here -- the left tail is longer. Think of it this way: "
                    "most values cluster toward the higher end, but there are occasional dips "
                    "that drag the mean below the median.")
else:
    st.info("No data available for the selected filters.")

# ── Code example ─────────────────────────────────────────────────────────────
code_example(
    """import pandas as pd

# Per-city descriptive stats
df.groupby("city")["temperature_c"].describe()

# Skewness and kurtosis
df.groupby("city")["temperature_c"].agg(["skew"])
df.groupby("city")["temperature_c"].apply(pd.Series.kurtosis)

# IQR
Q1 = df["temperature_c"].quantile(0.25)
Q3 = df["temperature_c"].quantile(0.75)
IQR = Q3 - Q1
"""
)

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "Which measure of central tendency is most robust to outliers?",
    ["Mean", "Median", "Standard Deviation", "Variance"],
    correct_idx=1,
    explanation="The median only cares about the middle value. You could replace the largest "
                "observation with a trillion and the median wouldn't budge. The mean, on the "
                "other hand, would completely lose its mind.",
    key="ch2_quiz1",
)

quiz(
    "A distribution with positive skewness has:",
    [
        "A longer left tail",
        "A longer right tail",
        "Equal tails",
        "No tails",
    ],
    correct_idx=1,
    explanation="Positive skew means the right tail (higher values) stretches out further. "
                "Think of income distributions: most people earn modest amounts, but a few earn enormously, "
                "pulling that right tail way out.",
    key="ch2_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "The mean is sensitive to outliers; the median is robust. Use both and compare -- if they diverge a lot, your data is probably skewed.",
    "Standard deviation and IQR both measure spread, but IQR ignores extremes. When in doubt about outliers, IQR is your friend.",
    "Skewness tells you which tail is longer; kurtosis tells you how heavy the tails are (not peakedness, despite the persistent myth).",
    "Never look at summary statistics in isolation. Always pair them with a histogram -- numbers lie by omission, pictures less so.",
    "Dallas temperature has high variability (large std); LA has low variability (small std). Same average, totally different lived experience.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 1: Exploring the Dataset",
    prev_page="01_Exploring_the_Dataset.py",
    next_label="Ch 3: Probability Distributions",
    next_page="03_Probability_Distributions.py",
)
