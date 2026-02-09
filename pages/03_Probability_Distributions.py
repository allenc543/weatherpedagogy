"""Chapter 3: Probability Distributions -- Normal, log-normal, PDF/CDF, QQ-plots, CLT."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as sp_stats

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, color_map
from utils.constants import (
    CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS, FEATURE_UNITS,
)
from utils.stats_helpers import descriptive_stats, normality_test
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(3, "Probability Distributions", part="I")
st.markdown(
    "Here's a question that sounds simple until you think about it: if I tell you "
    "it was 15 degrees C in Dallas at some random hour, is that *surprising*? To answer "
    "that, you need to know not just the average, but the full *distribution* -- the "
    "entire landscape of possible values and how likely each one is. That's what this "
    "chapter is about, and it turns out to be **foundational for basically everything** "
    "in statistics."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 3.1 The Normal Distribution ──────────────────────────────────────────────
st.header("3.1  The Normal (Gaussian) Distribution")

concept_box(
    "The Bell Curve That Runs the World",
    "The normal distribution is the famous bell curve, and it's defined by just two "
    "numbers: the mean (mu, where the peak sits) and the standard deviation (sigma, "
    "how wide it spreads). Why does it show up everywhere? Not because nature has a "
    "preference for bell shapes, but because of a deep mathematical result called the "
    "Central Limit Theorem, which we'll get to in section 3.6. For now, just know that "
    "whenever you add up lots of small independent effects, you tend to get a normal."
)

formula_box(
    "PDF of a Normal Distribution",
    r"\underbrace{f(x)}_{\text{probability density}} = \frac{1}{\underbrace{\sigma}_{\text{std dev}}\sqrt{2\pi}} \, e^{-\frac{(\overbrace{x}^{\text{observed value}} - \overbrace{\mu}^{\text{mean}})^2}{2\underbrace{\sigma^2}_{\text{variance}}}}",
    "mu = mean, sigma = standard deviation. The formula looks intimidating but it's "
    "really just saying: values near mu are likely, values far from mu are exponentially unlikely.",
)

# ── 3.2 Interactive: Is temperature normal within a city-month? ──────────────
st.header("3.2  Is Temperature Normal Within a City-Month?")

st.markdown(
    "Here's an interesting empirical question: if you take all the hourly temperature "
    "readings for a single city in a single month, does it actually look like a bell curve? "
    "Try different combinations below. You'll find that it often does -- not perfectly, "
    "but well enough that the normal distribution is a *useful approximation*. And in "
    "statistics, useful approximations are worth their weight in gold."
)

col1, col2 = st.columns(2)
with col1:
    city_sel = st.selectbox("City", sorted(fdf["city"].unique()), key="prob_city")
with col2:
    month_sel = st.selectbox(
        "Month",
        sorted(fdf["month"].unique()),
        format_func=lambda m: pd.Timestamp(2024, m, 1).strftime("%B"),
        key="prob_month",
    )

subset = fdf[(fdf["city"] == city_sel) & (fdf["month"] == month_sel)]["temperature_c"].dropna()

if len(subset) > 10:
    mu, sigma = subset.mean(), subset.std()

    # Histogram + fitted PDF overlay
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=subset, nbinsx=40, histnorm="probability density",
        name="Observed", marker_color=CITY_COLORS.get(city_sel, "#636EFA"),
        opacity=0.7,
    ))
    x_range = np.linspace(subset.min() - 2 * sigma, subset.max() + 2 * sigma, 300)
    pdf_vals = sp_stats.norm.pdf(x_range, mu, sigma)
    fig.add_trace(go.Scatter(
        x=x_range, y=pdf_vals, mode="lines", name="Fitted Normal PDF",
        line=dict(color="red", width=2),
    ))
    fig.update_layout(
        xaxis_title="Temperature (°C)", yaxis_title="Density",
        barmode="overlay",
    )
    apply_common_layout(fig, title=f"{city_sel} -- {pd.Timestamp(2024, month_sel, 1).strftime('%B')} Temperature", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Normality test
    stat, p = normality_test(subset.values)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Mean", f"{mu:.2f} °C")
    col_b.metric("Std Dev", f"{sigma:.2f} °C")
    col_c.metric("Shapiro-Wilk p-value", f"{p:.4f}")

    if p > 0.05:
        insight_box("The Shapiro-Wilk test says we can't reject normality (p > 0.05). "
                    "Translation from stats-speak: the data is consistent with being drawn "
                    "from a normal distribution. That doesn't *prove* it's normal -- you can "
                    "never prove a null hypothesis -- but the normal model is a reasonable fit.")
    else:
        warning_box("The Shapiro-Wilk test rejects normality here (p <= 0.05). The data "
                    "departs from a perfect bell curve. But here's the thing -- with enough data, "
                    "the Shapiro-Wilk test will reject normality for *any* real-world data, because "
                    "nothing is truly perfectly normal. The practical question is: is it close enough? "
                    "Look at the histogram and decide for yourself.")
else:
    st.info("Not enough data for the selected city and month.")

# ── 3.3 CDF ─────────────────────────────────────────────────────────────────
st.header("3.3  Cumulative Distribution Function (CDF)")

concept_box(
    "The CDF: Answering 'What Fraction Is Below X?'",
    "The CDF, F(x), answers a very natural question: what's the probability of seeing a "
    "value less than or equal to x? It rises from 0 to 1 -- impossible at the left edge, "
    "certain at the right. If you already know the PDF (the bell curve), the CDF is just "
    "the area under it up to point x. Or equivalently, the PDF is the derivative of the CDF."
)

if len(subset) > 10:
    sorted_vals = np.sort(subset)
    ecdf_y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    theoretical_cdf = sp_stats.norm.cdf(sorted_vals, mu, sigma)

    fig_cdf = go.Figure()
    fig_cdf.add_trace(go.Scatter(x=sorted_vals, y=ecdf_y, mode="lines",
                                  name="Empirical CDF", line=dict(color=CITY_COLORS.get(city_sel, "blue"))))
    fig_cdf.add_trace(go.Scatter(x=sorted_vals, y=theoretical_cdf, mode="lines",
                                  name="Theoretical Normal CDF", line=dict(color="red", dash="dash")))
    fig_cdf.update_layout(xaxis_title="Temperature (°C)", yaxis_title="Cumulative Probability")
    apply_common_layout(fig_cdf, title="Empirical vs Theoretical CDF", height=400)
    st.plotly_chart(fig_cdf, use_container_width=True)

# ── 3.4 QQ Plot ─────────────────────────────────────────────────────────────
st.header("3.4  QQ Plot")

concept_box(
    "QQ Plots: The Best Normality Diagnostic You're Not Using",
    "A Quantile-Quantile plot is clever: it lines up the quantiles of your data against "
    "the quantiles of a theoretical distribution (usually normal). If your data is actually "
    "normal, every point falls on a nice 45-degree line. Deviations from that line tell you "
    "*how* your data isn't normal -- S-curves mean skew, heavy tails curve away at the ends. "
    "It's much more informative than any single test statistic."
)

if len(subset) > 10:
    theoretical_q = sp_stats.norm.ppf(
        np.linspace(0.01, 0.99, len(subset)), mu, sigma
    )
    observed_q = np.sort(subset.values)

    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(
        x=theoretical_q, y=observed_q, mode="markers",
        marker=dict(size=3, color=CITY_COLORS.get(city_sel, "blue"), opacity=0.5),
        name="Data",
    ))
    min_val = min(theoretical_q.min(), observed_q.min())
    max_val = max(theoretical_q.max(), observed_q.max())
    fig_qq.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", name="Reference Line",
        line=dict(color="red", dash="dash"),
    ))
    fig_qq.update_layout(xaxis_title="Theoretical Quantiles", yaxis_title="Observed Quantiles")
    apply_common_layout(fig_qq, title="QQ Plot: Temperature vs Normal", height=450)
    st.plotly_chart(fig_qq, use_container_width=True)

# ── 3.5 Wind Speed: a right-skewed distribution ─────────────────────────────
st.header("3.5  Wind Speed: A Case Where Normal Fails")

st.markdown(
    "Not everything is normal, and wind speed is a great example. Think about it: "
    "wind speed can't be negative (you can't blow at -5 km/h), and most hours are "
    "relatively calm, but occasionally you get strong gusts. This creates a distribution "
    "that piles up near zero and has a long right tail. The normal distribution, with "
    "its perfect symmetry, is the wrong model here. Enter the **log-normal** and **Weibull** "
    "distributions, which were basically *invented* for data like this."
)

wind_data = fdf.loc[fdf["city"] == city_sel, "wind_speed_kmh"].dropna()
if len(wind_data) > 10:
    wind_mu, wind_sigma = wind_data.mean(), wind_data.std()
    wind_skew = wind_data.skew()

    fig_wind = go.Figure()
    fig_wind.add_trace(go.Histogram(
        x=wind_data, nbinsx=50, histnorm="probability density",
        name="Observed", marker_color=CITY_COLORS.get(city_sel, "teal"), opacity=0.7,
    ))
    # Fit log-normal
    log_wind = np.log(wind_data[wind_data > 0])
    ln_mu, ln_sigma = log_wind.mean(), log_wind.std()
    x_range_w = np.linspace(0.01, wind_data.max(), 300)
    lognorm_pdf = sp_stats.lognorm.pdf(x_range_w, s=ln_sigma, scale=np.exp(ln_mu))
    fig_wind.add_trace(go.Scatter(
        x=x_range_w, y=lognorm_pdf, mode="lines", name="Fitted Log-Normal PDF",
        line=dict(color="red", width=2),
    ))
    fig_wind.update_layout(xaxis_title="Wind Speed (km/h)", yaxis_title="Density")
    apply_common_layout(fig_wind, title=f"{city_sel} -- Wind Speed Distribution (Skewness = {wind_skew:.2f})", height=450)
    st.plotly_chart(fig_wind, use_container_width=True)

    insight_box(
        f"Wind speed for {city_sel} has a skewness of {wind_skew:.2f}. What does that look "
        "like in practice? Most hours are pretty calm, but there's a long tail of occasional "
        "strong winds dragging the distribution to the right. If you used a normal model here, "
        "it would predict negative wind speeds with non-trivial probability, which is... not how wind works."
    )

# ── 3.6 Central Limit Theorem Demonstration ─────────────────────────────────
st.header("3.6  The Central Limit Theorem (CLT)")

concept_box(
    "The Central Limit Theorem: Why Normal Distributions Are Everywhere",
    "Here's the weird and wonderful thing about the Central Limit Theorem: take *any* "
    "distribution -- skewed, bimodal, uniform, whatever -- and start averaging random "
    "samples from it. The distribution of those averages will approach a normal distribution "
    "as sample size grows. It doesn't matter what the original distribution looked like. "
    "This is, arguably, the single most important theorem in all of statistics, and it's why "
    "the normal distribution shows up in places where you'd never expect it."
)

formula_box(
    "Standard Error of the Mean",
    r"\underbrace{SE}_{\text{standard error}} = \frac{\underbrace{\sigma}_{\text{population std dev}}}{\sqrt{\underbrace{n}_{\text{sample size}}}}",
    "As n increases, SE shrinks. This means the sampling distribution of the mean gets "
    "narrower and narrower -- your averages become more and more precise. But notice the "
    "square root: to halve the SE, you need to quadruple n. Diminishing returns are real.",
)

st.markdown(
    "Let's watch the CLT in action. Pick a feature -- even something skewed like "
    "wind speed -- and watch how the distribution of sample means becomes bell-shaped "
    "as the sample size grows. It's almost magical."
)

clt_feature = st.selectbox(
    "Feature for CLT demo",
    FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    key="clt_feature",
)

pop = fdf[clt_feature].dropna().values
n_simulations = st.slider("Number of samples to draw", 100, 2000, 500, step=100, key="clt_nsim")

sample_sizes = [5, 15, 30, 100]
rng = np.random.RandomState(42)

fig_clt = go.Figure()
for ss in sample_sizes:
    sample_means = [rng.choice(pop, size=ss, replace=True).mean() for _ in range(n_simulations)]
    fig_clt.add_trace(go.Histogram(
        x=sample_means, nbinsx=40, histnorm="probability density",
        name=f"n = {ss}", opacity=0.5,
    ))

fig_clt.update_layout(
    barmode="overlay",
    xaxis_title=f"Sample Mean of {FEATURE_LABELS[clt_feature]}",
    yaxis_title="Density",
)
apply_common_layout(fig_clt, title="CLT: Sampling Distributions at Different Sample Sizes", height=500)
st.plotly_chart(fig_clt, use_container_width=True)

insight_box(
    "This is the CLT doing its thing right before your eyes. At n=5, the distribution "
    "of sample means is still lumpy and wide. By n=30, it's starting to look bell-shaped. "
    "At n=100, it's practically textbook normal -- even if the original data was skewed. "
    "This is why 'n=30' has become a folk threshold for 'large enough sample' in introductory "
    "statistics, though the actual number you need depends on how skewed your original data is."
)

# ── Code example ─────────────────────────────────────────────────────────────
code_example(
    """from scipy import stats
import numpy as np

# Fit a normal distribution to temperature data
mu, sigma = data.mean(), data.std()

# Compute the PDF
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
pdf = stats.norm.pdf(x, mu, sigma)

# Shapiro-Wilk normality test
stat, p = stats.shapiro(data[:5000])
print(f"Shapiro-Wilk statistic={stat:.4f}, p-value={p:.4f}")

# QQ plot (using scipy)
from scipy.stats import probplot
probplot(data, dist="norm", plot=plt)

# CLT simulation
sample_means = [np.random.choice(data, size=30).mean() for _ in range(1000)]
"""
)

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "What does the Central Limit Theorem guarantee?",
    [
        "Individual data points become normally distributed with larger samples",
        "The distribution of sample means approaches a normal distribution as n increases",
        "The population standard deviation decreases with more data",
        "All real-world data follows a normal distribution",
    ],
    correct_idx=1,
    explanation="The CLT is specifically about the distribution of the *sample mean*, not individual "
                "data points. Your raw data can be as weird as it wants -- the averages will still "
                "converge to normal. This is the key insight people often miss.",
    key="ch3_quiz1",
)

quiz(
    "On a QQ plot, what does it mean when points follow the diagonal line?",
    [
        "The data has high variance",
        "The data matches the theoretical distribution",
        "The data is skewed",
        "The sample size is too small",
    ],
    correct_idx=1,
    explanation="Points on the 45-degree line mean your observed quantiles match the theoretical "
                "quantiles almost exactly. When they deviate, *where* they deviate tells you *how* -- "
                "which is why QQ plots are so much more informative than pass/fail normality tests.",
    key="ch3_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "The normal distribution is defined by just two parameters (mean and std) and appears everywhere thanks to the CLT.",
    "Temperature within a single city-month is often approximately normal -- not perfectly, but close enough to be useful.",
    "Wind speed is right-skewed and needs a different model (log-normal or Weibull). Always check, never assume normal.",
    "The CDF answers 'what fraction of values fall below x?' -- simple question, surprisingly powerful tool.",
    "QQ plots are the best visual diagnostic for normality. They tell you not just *if* the data is non-normal, but *how*.",
    "The CLT explains why the normal distribution is central to statistics: averages of anything tend to be normal.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 2: Descriptive Statistics",
    prev_page="02_Descriptive_Statistics.py",
    next_label="Ch 4: Sampling & Estimation",
    next_page="04_Sampling_and_Estimation.py",
)
