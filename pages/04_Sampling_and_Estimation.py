"""Chapter 4: Sampling & Estimation -- Population vs sample, SE, Law of Large Numbers."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, color_map
from utils.constants import (
    CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS, FEATURE_UNITS,
)
from utils.stats_helpers import descriptive_stats, bootstrap_ci
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(4, "Sampling & Estimation", part="I")
st.markdown(
    "Here's the fundamental problem of statistics, the one that makes the whole field "
    "necessary: you almost never get to see the full picture. You want to know the "
    "average temperature in Dallas, but you only have a thermometer on your porch for a "
    "few weeks. You want to know how Americans will vote, but you can only call 1,000 of "
    "them. In this chapter, we'll treat our complete weather dataset as the **population** "
    "and draw samples from it, so you can *see* how sampling works -- how much your "
    "estimates wiggle around, why bigger samples are better, and exactly *how much* better."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 4.1 Population vs Sample ────────────────────────────────────────────────
st.header("4.1  Population vs Sample")

concept_box(
    "The Population/Sample Distinction (It Matters More Than You Think)",
    "<b>Population</b>: the complete set of everything you care about. Here, that's our "
    "entire weather dataset -- every hourly observation across all cities.<br>"
    "<b>Sample</b>: a subset you actually get to look at. In the real world, you sample "
    "because measuring everything is too expensive, too slow, or literally impossible.<br>"
    "<b>Parameter</b>: a fixed number describing the population (e.g., the true mean "
    "temperature mu). You usually don't know it.<br>"
    "<b>Statistic</b>: a number you compute from your sample (e.g., the sample mean "
    "x-bar). It's your best guess at the parameter, but it comes with uncertainty."
)

# ── 4.2 Interactive sampling ────────────────────────────────────────────────
st.header("4.2  Draw a Random Sample")

feature = st.selectbox(
    "Feature",
    FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    key="samp_feature",
)
city_sel = st.selectbox("City", sorted(fdf["city"].unique()), key="samp_city")

pop_data = fdf.loc[fdf["city"] == city_sel, feature].dropna()
pop_mean = pop_data.mean()
pop_std = pop_data.std()

st.write(f"**Population** (all {len(pop_data):,} observations for {city_sel}):")
st.write(f"Population mean (mu) = **{pop_mean:.3f}** {FEATURE_UNITS.get(feature, '')}")
st.write(f"Population std (sigma) = **{pop_std:.3f}** {FEATURE_UNITS.get(feature, '')}")

sample_size = st.slider(
    "Sample size (n)", min_value=5, max_value=min(2000, len(pop_data)),
    value=50, step=5, key="samp_size",
)
seed_val = st.number_input("Random seed (change to draw a different sample)", value=42, step=1, key="samp_seed")

rng = np.random.RandomState(int(seed_val))
sample = rng.choice(pop_data.values, size=sample_size, replace=False)
sample_mean = sample.mean()
sample_std = sample.std(ddof=1)
se = pop_std / np.sqrt(sample_size)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sample Mean", f"{sample_mean:.3f}", delta=f"{sample_mean - pop_mean:.3f} from mu")
col2.metric("Sample Std", f"{sample_std:.3f}")
col3.metric("Standard Error", f"{se:.3f}")
col4.metric("Sample Size", sample_size)

# Histogram of sample vs population
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=pop_data, nbinsx=60, histnorm="probability density",
    name="Population", marker_color="lightgrey", opacity=0.5,
))
fig.add_trace(go.Histogram(
    x=sample, nbinsx=30, histnorm="probability density",
    name=f"Sample (n={sample_size})", marker_color=CITY_COLORS.get(city_sel, "blue"),
    opacity=0.7,
))
fig.add_vline(x=pop_mean, line_dash="solid", line_color="black",
              annotation_text="Pop Mean", annotation_position="top right")
fig.add_vline(x=sample_mean, line_dash="dash", line_color="red",
              annotation_text="Sample Mean", annotation_position="top left")
fig.update_layout(barmode="overlay", xaxis_title=FEATURE_LABELS.get(feature, feature),
                  yaxis_title="Density")
apply_common_layout(fig, title="Population vs Sample Distribution", height=450)
st.plotly_chart(fig, use_container_width=True)

formula_box(
    "Standard Error of the Mean",
    r"\underbrace{SE}_{\text{standard error}} = \frac{\underbrace{\sigma}_{\text{population std dev}}}{\sqrt{\underbrace{n}_{\text{sample size}}}}",
    "The standard error tells you how much your sample mean would bounce around if "
    "you kept drawing new samples of the same size. Small SE = precise estimate. "
    "Large SE = your sample mean could easily be far from the truth.",
)

# ── 4.3 Law of Large Numbers ────────────────────────────────────────────────
st.header("4.3  Law of Large Numbers")

concept_box(
    "The Law of Large Numbers: Patience Pays Off",
    "The Law of Large Numbers says something that feels obvious but is mathematically "
    "profound: as you collect more and more data, your sample mean converges to the "
    "population mean. With 10 observations you might be way off. With 10,000, you'll "
    "be very close. The key word is 'converges' -- it's not guaranteed for any particular "
    "finite sample, but the trend is inexorable."
)

st.markdown(
    "Watch the running sample mean below. At first it jumps around wildly -- with just "
    "a few observations, a single extreme value can push the average far from the truth. "
    "But as more observations pile up, each individual value matters less, and the "
    "average settles down."
)

max_n = min(3000, len(pop_data))
running_sample = rng.choice(pop_data.values, size=max_n, replace=False)
running_means = np.cumsum(running_sample) / np.arange(1, max_n + 1)

fig_lln = go.Figure()
fig_lln.add_trace(go.Scatter(
    x=np.arange(1, max_n + 1), y=running_means,
    mode="lines", name="Running Mean",
    line=dict(color=CITY_COLORS.get(city_sel, "blue")),
))
fig_lln.add_hline(y=pop_mean, line_dash="dash", line_color="red",
                  annotation_text=f"Population Mean = {pop_mean:.2f}")
fig_lln.update_layout(
    xaxis_title="Number of Observations",
    yaxis_title=f"Running Mean ({FEATURE_UNITS.get(feature, '')})",
)
apply_common_layout(fig_lln, title="Law of Large Numbers: Running Mean Converges", height=450)
st.plotly_chart(fig_lln, use_container_width=True)

insight_box(
    "That wild early behavior and eventual convergence is the LLN in its natural habitat. "
    "The first 10-20 observations produce a chaotic running mean -- you'd be forgiven for "
    "doubting convergence at that point. But math is relentless. By a few hundred observations, "
    "the mean barely moves."
)

# ── 4.4 Repeated Sampling Demonstration ─────────────────────────────────────
st.header("4.4  Repeated Sampling: The Sampling Distribution of the Mean")

st.markdown(
    "Here's a thought experiment that's central to understanding statistics: imagine you "
    "draw a sample, compute its mean, then throw it away and draw another. Repeat this "
    "hundreds of times. What does the distribution of all those sample means look like? "
    "This is called the **sampling distribution**, and it's the key to understanding "
    "uncertainty in estimation."
)

n_reps = st.slider("Number of repeated samples", 100, 2000, 500, step=100, key="rep_nreps")
rep_size = st.slider("Each sample's size (n)", 5, 500, 30, step=5, key="rep_size")

rng2 = np.random.RandomState(42)
sample_means = [rng2.choice(pop_data.values, size=rep_size, replace=True).mean()
                for _ in range(n_reps)]

theoretical_se = pop_std / np.sqrt(rep_size)
observed_se = np.std(sample_means, ddof=1)

fig_samp = go.Figure()
fig_samp.add_trace(go.Histogram(
    x=sample_means, nbinsx=50, histnorm="probability density",
    name="Sample Means", marker_color=CITY_COLORS.get(city_sel, "steelblue"),
    opacity=0.7,
))
# Overlay theoretical normal
from scipy import stats as sp_stats
x_range = np.linspace(min(sample_means), max(sample_means), 200)
pdf_vals = sp_stats.norm.pdf(x_range, pop_mean, theoretical_se)
fig_samp.add_trace(go.Scatter(
    x=x_range, y=pdf_vals, mode="lines", name="Theoretical Normal",
    line=dict(color="red", width=2),
))
fig_samp.update_layout(
    xaxis_title=f"Sample Mean ({FEATURE_UNITS.get(feature, '')})",
    yaxis_title="Density",
    barmode="overlay",
)
apply_common_layout(fig_samp, title=f"Sampling Distribution (n={rep_size}, {n_reps} samples)", height=450)
st.plotly_chart(fig_samp, use_container_width=True)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Population Mean", f"{pop_mean:.3f}")
col_b.metric("Theoretical SE", f"{theoretical_se:.3f}")
col_c.metric("Observed SE of Sample Means", f"{observed_se:.3f}")

insight_box(
    f"The theoretical SE ({theoretical_se:.3f}) and observed SE ({observed_se:.3f}) "
    "are remarkably close. This isn't a coincidence -- it's confirmation that the formula "
    "SE = sigma / sqrt(n) actually works. The math predicts reality. When that happens in "
    "statistics, it's a good sign you're doing something right."
)

# ── 4.5 How Sample Size Affects SE ──────────────────────────────────────────
st.header("4.5  Standard Error vs Sample Size")

sizes = np.arange(5, 501, 5)
se_vals = pop_std / np.sqrt(sizes)

fig_se = go.Figure()
fig_se.add_trace(go.Scatter(
    x=sizes, y=se_vals, mode="lines",
    line=dict(color=CITY_COLORS.get(city_sel, "purple"), width=2),
    name="SE = sigma / sqrt(n)",
))
fig_se.update_layout(xaxis_title="Sample Size (n)", yaxis_title="Standard Error")
apply_common_layout(fig_se, title="Standard Error Decreases with Sample Size", height=400)
st.plotly_chart(fig_se, use_container_width=True)

warning_box(
    "Notice the shape of this curve: it drops steeply at first, then flattens out. "
    "Going from n=5 to n=50 is a huge improvement. Going from n=500 to n=5000 barely "
    "matters. Specifically, doubling your sample size does NOT halve the SE. You need "
    "4x the sample size to halve SE, because SE scales with 1/sqrt(n). This has real "
    "implications for how much data you should collect -- at some point, more data is "
    "just not worth the cost."
)

# ── Code example ─────────────────────────────────────────────────────────────
code_example(
    """import numpy as np

population = df.loc[df["city"] == "Dallas", "temperature_c"].values
pop_mean = population.mean()
pop_std = population.std()

# Draw a random sample
sample = np.random.choice(population, size=50, replace=False)
sample_mean = sample.mean()

# Standard error
SE = pop_std / np.sqrt(50)

# Repeated sampling: distribution of sample means
sample_means = [np.random.choice(population, size=50).mean()
                for _ in range(1000)]
print("Mean of sample means:", np.mean(sample_means))
print("SE of sample means:", np.std(sample_means))
"""
)

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "To halve the standard error, you must multiply the sample size by:",
    ["2", "4", "8", "16"],
    correct_idx=1,
    explanation="SE = sigma/sqrt(n). To get SE/2 you need sqrt(4n) = 2*sqrt(n), which means "
                "n must be multiplied by 4. This is the single most important practical fact about "
                "sampling: the square root makes increasing precision increasingly expensive.",
    key="ch4_quiz1",
)

quiz(
    "The Law of Large Numbers states that as n increases:",
    [
        "The standard deviation increases",
        "The population mean changes",
        "The sample mean converges to the population mean",
        "The data becomes normally distributed",
    ],
    correct_idx=2,
    explanation="The LLN is specifically about convergence of the sample mean to the population mean. "
                "It doesn't change the population, it doesn't reshape the data -- it just says your "
                "estimate gets better and better. Simple but powerful.",
    key="ch4_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "A sample statistic is your best guess at the population parameter -- but it comes with uncertainty, and that uncertainty has a name: standard error.",
    "SE = sigma/sqrt(n) tells you exactly how precise your sample mean is. Larger n = smaller SE = better estimate.",
    "The Law of Large Numbers guarantees convergence of the sample mean to the population mean. Math keeps its promises.",
    "The sampling distribution of the mean is approximately normal (CLT) regardless of the original data's shape, as long as n is large enough.",
    "Quadrupling the sample size halves the standard error. Diminishing returns are baked into the math via the square root.",
    "Always think about whether your data is a sample or a population. The statistical tools you use depend on the answer.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 3: Probability Distributions",
    prev_page="03_Probability_Distributions.py",
    next_label="Ch 5: Histograms & Density",
    next_page="05_Histograms_and_Density.py",
)
