"""Chapter 5: Histograms & Density -- Bin width effects, KDE, frequency vs density."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as sp_stats

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import histogram_chart, apply_common_layout, color_map
from utils.constants import (
    CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS, FEATURE_UNITS,
    SEASON_ORDER,
)
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(5, "Histograms & Density", part="II")
st.markdown(
    "We've spent four chapters talking about data in the abstract -- means, standard "
    "deviations, distributions. Now it's time to actually *look* at things. And the "
    "most fundamental way to look at continuous data is the histogram. It sounds simple -- "
    "just count how many values fall into each bucket, right? -- but there are some "
    "**surprisingly tricky choices** hiding inside this simplicity."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 5.1 Theory ──────────────────────────────────────────────────────────────
st.header("5.1  Anatomy of a Histogram")

concept_box(
    "What Is a Histogram, Really?",
    "A histogram takes the range of your variable, chops it into equal-width <b>bins</b>, "
    "and counts how many observations land in each one. The height of each bar shows either "
    "the <b>frequency</b> (raw count) or the <b>density</b> (proportion per unit width). "
    "It's the data scientist's bread and butter, the first chart you reach for when you "
    "want to understand a variable."
)

concept_box(
    "Frequency vs Density (This Distinction Matters)",
    "<b>Frequency</b>: the raw count in each bin. All bar heights add up to N (total "
    "observations). Simple to interpret but impossible to compare across datasets of "
    "different sizes.<br>"
    "<b>Density</b>: each bar's *area* (not height!) equals the proportion of data in "
    "that bin. Total area sums to 1. This lets you overlay a probability density function "
    "on top -- like that normal curve from Chapter 3 -- and lets you meaningfully compare "
    "distributions with different sample sizes."
)

# ── 5.2 Interactive: bin width ───────────────────────────────────────────────
st.header("5.2  How Bins Change the Story")

st.markdown(
    "Here's where it gets interesting. The *same data* can tell very different stories "
    "depending on how many bins you use. Too few bins and you lose all the structure. "
    "Too many bins and you're basically plotting noise. Try the slider below and watch "
    "how the shape changes -- it's a nice reminder that visualization choices aren't neutral."
)

col_a, col_b = st.columns(2)
with col_a:
    city_sel = st.selectbox("City", sorted(fdf["city"].unique()), key="hist_city")
with col_b:
    feat_sel = st.selectbox(
        "Feature", FEATURE_COLS,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
        key="hist_feat",
    )

nbins = st.slider("Number of bins", 5, 200, 50, step=5, key="hist_nbins")

city_data = fdf.loc[fdf["city"] == city_sel, feat_sel].dropna()

if len(city_data) > 0:
    hist_mode = st.radio(
        "Y-axis mode", ["Frequency (count)", "Density"],
        horizontal=True, key="hist_mode",
    )
    histnorm = "" if hist_mode.startswith("Freq") else "probability density"

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=city_data, nbinsx=nbins,
        histnorm=histnorm,
        marker_color=CITY_COLORS.get(city_sel, "steelblue"),
        opacity=0.75, name="Histogram",
    ))
    fig.update_layout(
        xaxis_title=FEATURE_LABELS.get(feat_sel, feat_sel),
        yaxis_title="Count" if hist_mode.startswith("Freq") else "Density",
    )
    apply_common_layout(fig, title=f"{city_sel} -- {FEATURE_LABELS[feat_sel]} ({nbins} bins)", height=450)
    st.plotly_chart(fig, use_container_width=True)

warning_box(
    "With fewer than ~10 bins, you can easily miss multi-modality (two humps that should "
    "be visible). With more than ~150 bins, you start seeing random spikes that mean nothing. "
    "There are mathematical rules for choosing bins -- Sturges' rule (k = 1 + log2(n)) and "
    "Freedman-Diaconis (bin width = 2 * IQR / n^(1/3)) -- but honestly, the best approach "
    "is to try a few values and see what reveals the most structure."
)

# ── 5.3 KDE overlay ─────────────────────────────────────────────────────────
st.header("5.3  Kernel Density Estimation (KDE)")

concept_box(
    "KDE: The Smooth Alternative to Histograms",
    "Here's the idea behind KDE: instead of shoving your data into bins, place a smooth "
    "little bell curve (a 'kernel') at each data point, then add all those curves up. "
    "The result is a continuous density estimate. The <b>bandwidth</b> parameter controls "
    "how wide each little kernel is -- small bandwidth makes the curve wiggly and responsive "
    "to local detail, large bandwidth smooths everything out. It's the continuous analog of "
    "bin width."
)

show_kde = st.checkbox("Overlay KDE curve", value=True, key="hist_kde_toggle")
bw_factor = st.slider("KDE bandwidth multiplier", 0.1, 3.0, 1.0, step=0.1, key="hist_bw")

if len(city_data) > 0:
    fig_kde = go.Figure()
    fig_kde.add_trace(go.Histogram(
        x=city_data, nbinsx=nbins, histnorm="probability density",
        marker_color=CITY_COLORS.get(city_sel, "steelblue"),
        opacity=0.5, name="Histogram",
    ))

    if show_kde:
        kde = sp_stats.gaussian_kde(city_data, bw_method=bw_factor * sp_stats.gaussian_kde(city_data).factor)
        x_range = np.linspace(city_data.min(), city_data.max(), 500)
        fig_kde.add_trace(go.Scatter(
            x=x_range, y=kde(x_range), mode="lines",
            name="KDE", line=dict(color="red", width=2),
        ))

    fig_kde.update_layout(
        xaxis_title=FEATURE_LABELS.get(feat_sel, feat_sel),
        yaxis_title="Density",
        barmode="overlay",
    )
    apply_common_layout(fig_kde, title=f"Histogram + KDE ({city_sel})", height=450)
    st.plotly_chart(fig_kde, use_container_width=True)

    insight_box(
        "Play with the bandwidth slider. A small value (< 0.5) makes the KDE faithfully "
        "trace every bump in the data -- useful for spotting multi-modality, but also "
        "prone to fitting noise. A large value (> 2.0) smooths out real features. "
        "The default (1.0) is usually a reasonable compromise, but 'reasonable' depends "
        "on what question you're asking."
    )

# ── 5.4 Seasonal temperature shifts ─────────────────────────────────────────
st.header("5.4  Seasonal Temperature Shifts")

st.markdown(
    "Here's a satisfying thing you can do with overlapping histograms: show how the "
    "distribution of temperature *moves* across seasons. Summer doesn't just mean "
    "'higher average temperature' -- it means the entire distribution slides to the right. "
    "And you can see it."
)

season_data = fdf[fdf["city"] == city_sel][["temperature_c", "season"]].dropna()

if len(season_data) > 0:
    fig_season = go.Figure()
    for season in SEASON_ORDER:
        s_data = season_data.loc[season_data["season"] == season, "temperature_c"]
        if len(s_data) > 0:
            fig_season.add_trace(go.Histogram(
                x=s_data, nbinsx=40, histnorm="probability density",
                name=season, opacity=0.5,
            ))
    fig_season.update_layout(
        barmode="overlay",
        xaxis_title="Temperature (°C)",
        yaxis_title="Density",
    )
    apply_common_layout(fig_season, title=f"Temperature Distribution by Season -- {city_sel}", height=450)
    st.plotly_chart(fig_season, use_container_width=True)

    insight_box(
        "Summer and winter are clearly separated -- they barely overlap. Spring and fall "
        "are the transition seasons, and their distributions overlap with both neighbors. "
        "If you ever wondered why spring weather feels so unpredictable, this is part of "
        "the answer: you're living in the overlap zone between two different distributions."
    )

# ── 5.5 Wind speed skew ─────────────────────────────────────────────────────
st.header("5.5  Wind Speed: Skewness You Can See")

st.markdown(
    "We talked about skewness as a number in Chapter 2. Now let's *see* what it "
    "looks like. Wind speed is always non-negative (you can't have negative wind) and "
    "most hours are relatively calm, so the distribution piles up on the left with a "
    "long tail stretching right. Compare cities below -- some are windier than others, "
    "but they're all right-skewed."
)

cities_to_compare = st.multiselect(
    "Cities to compare",
    sorted(fdf["city"].unique()),
    default=["Dallas", "Los Angeles"],
    key="hist_wind_cities",
)

wind_compare = fdf[fdf["city"].isin(cities_to_compare)][["wind_speed_kmh", "city"]].dropna()
if len(wind_compare) > 0:
    fig_wind = go.Figure()
    for city in cities_to_compare:
        w = wind_compare.loc[wind_compare["city"] == city, "wind_speed_kmh"]
        fig_wind.add_trace(go.Histogram(
            x=w, nbinsx=50, histnorm="probability density",
            name=city, marker_color=CITY_COLORS.get(city, None),
            opacity=0.5,
        ))
    fig_wind.update_layout(
        barmode="overlay",
        xaxis_title="Wind Speed (km/h)",
        yaxis_title="Density",
    )
    apply_common_layout(fig_wind, title="Wind Speed Distribution by City", height=450)
    st.plotly_chart(fig_wind, use_container_width=True)

# ── 5.6 Multi-city comparison ───────────────────────────────────────────────
st.header("5.6  Multi-City Feature Comparison")

feat_compare = st.selectbox(
    "Feature",
    FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    key="hist_multi_feat",
)

fig_multi = histogram_chart(fdf, x=feat_compare, nbins=60,
                            title=f"{FEATURE_LABELS[feat_compare]} -- All Selected Cities")
st.plotly_chart(fig_multi, use_container_width=True)

# ── Code example ─────────────────────────────────────────────────────────────
code_example(
    """import plotly.express as px
from scipy import stats

# Basic histogram
fig = px.histogram(df, x="temperature_c", nbins=50,
                   color="city", barmode="overlay", opacity=0.6)
fig.show()

# KDE with scipy
from scipy.stats import gaussian_kde
kde = gaussian_kde(data, bw_method=0.3)
x = np.linspace(data.min(), data.max(), 500)
density = kde(x)

# Freedman-Diaconis bin width
IQR = np.percentile(data, 75) - np.percentile(data, 25)
bin_width = 2 * IQR / len(data)**(1/3)
n_bins = int((data.max() - data.min()) / bin_width)
"""
)

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "In a density histogram, what do the bar areas sum to?",
    ["The sample size N", "1", "The mean", "The standard deviation"],
    correct_idx=1,
    explanation="In a density histogram, the total area under the bars equals 1. That's what "
                "makes it comparable to a probability density function -- you can overlay a theoretical "
                "distribution on top and the scales actually match.",
    key="ch5_quiz1",
)

quiz(
    "What effect does increasing KDE bandwidth have?",
    [
        "Makes the curve more wiggly",
        "Makes the curve smoother",
        "Changes the area under the curve",
        "Shifts the curve to the right",
    ],
    correct_idx=1,
    explanation="Higher bandwidth spreads each kernel wider, smoothing out local bumps. "
                "Think of it like looking at the data through increasingly blurry glasses -- "
                "you lose the fine detail but the big-picture shape stays.",
    key="ch5_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "Bin width dramatically affects what a histogram reveals. Always try multiple values before committing to one.",
    "Density histograms (area = 1) are the right choice when comparing distributions of different sizes or overlaying theoretical PDFs.",
    "KDE gives you a smooth density estimate. The bandwidth parameter is the continuous cousin of bin width -- same tradeoffs, same need for judgment.",
    "Seasonal temperature histograms visibly slide left (winter) and right (summer). Spring and fall live in the overlap zone.",
    "Wind speed is right-skewed across all cities -- most hours are calm, with occasional strong winds creating the tail.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 4: Sampling & Estimation",
    prev_page="04_Sampling_and_Estimation.py",
    next_label="Ch 6: Scatter Plots & Relationships",
    next_page="06_Scatter_Plots_and_Relationships.py",
)
