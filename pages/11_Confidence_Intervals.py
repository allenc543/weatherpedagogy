"""Chapter 11: Confidence Intervals — CI width, bootstrap CI, confidence level interpretation."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, color_map, histogram_chart
from utils.stats_helpers import descriptive_stats, bootstrap_ci
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)
from utils.constants import CITY_LIST, FEATURE_COLS, FEATURE_LABELS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
df = load_data()
fdf = sidebar_filters(df)

chapter_header(11, "Confidence Intervals", part="III")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "What Is a Confidence Interval, Really?",
    "Imagine you're trying to figure out the average temperature in Dallas, but you can only "
    "measure it on 500 random days. You get a number -- say, 19.3 C -- but you know that's "
    "not <em>exactly</em> right. If you'd picked 500 different days, you'd have gotten something "
    "slightly different. A confidence interval is your way of being honest about this: instead of "
    "saying 'the true mean is 19.3,' you say 'the true mean is <b>probably somewhere between "
    "18.5 and 20.1</b>.' That range is your CI, and it communicates something a point estimate "
    "never can: how much you should trust the number you got.",
)

formula_box(
    "CI for the Mean (Normal approximation)",
    r"\bar{x} \;\pm\; z_{\alpha/2} \cdot \frac{s}{\sqrt{n}}",
    "where x-bar is the sample mean, s is the sample standard deviation, n is the sample size, "
    "and z is the critical value from the standard normal distribution.",
)

st.markdown("### Three things that make a confidence interval wider or narrower")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**1. Sample size (n)**")
    st.markdown("More data means more precision. Double your sample size and the interval shrinks -- not by half, but by a factor of sqrt(2), because statistics is never as generous as you'd like.")
with col2:
    st.markdown("**2. Variability (s)**")
    st.markdown("If your data is all over the place, it's harder to pin down the mean. Dallas temperatures in July are pretty consistent; across all seasons, they're wildly variable. The CI reflects this.")
with col3:
    st.markdown("**3. Confidence level**")
    st.markdown("Want to be 99% sure instead of 95%? That certainty costs you. The interval has to widen to maintain its promise. It's the statistical version of 'you can't have it all.'")

warning_box(
    "Here's the thing that trips up approximately everyone, including people who should know "
    "better: a 95% CI does NOT mean there's a 95% probability the true mean is inside this "
    "particular interval. The true mean is a fixed number sitting out there in reality. It's "
    "either in the interval or it isn't. The '95%' refers to the <em>procedure</em>: if you "
    "repeated this whole sampling-and-interval-computing process 100 times, about 95 of those "
    "intervals would contain the true mean. It's a statement about the method, not about any "
    "single interval. (Yes, this is confusing. No, there's no way around it in frequentist statistics.)",
)

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive Demo: CI Width Explorer
# ---------------------------------------------------------------------------
st.subheader("Play With It: CI Width Explorer")

col_left, col_right = st.columns([1, 2])

with col_left:
    ci_city = st.selectbox("City", CITY_LIST, key="ci_city")
    ci_feature = st.selectbox(
        "Feature",
        FEATURE_COLS,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
        key="ci_feature",
    )
    conf_level = st.slider("Confidence level (%)", 80, 99, 95, key="ci_conf")
    sample_size = st.slider("Sample size", 10, 5000, 500, step=10, key="ci_n")

city_data = fdf[fdf["city"] == ci_city][ci_feature].dropna().values

# True population mean (from all data for that city)
pop_data = df[df["city"] == ci_city][ci_feature].dropna().values
pop_mean = pop_data.mean()

# Draw a random sample
rng = np.random.RandomState(42)
sample = rng.choice(city_data, size=min(sample_size, len(city_data)), replace=False)
sample_mean = sample.mean()
sample_std = sample.std(ddof=1)
n = len(sample)

from scipy import stats as sp_stats
z = sp_stats.norm.ppf(1 - (1 - conf_level / 100) / 2)
margin = z * sample_std / np.sqrt(n)
ci_lower = sample_mean - margin
ci_upper = sample_mean + margin

with col_right:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sample, nbinsx=40, name="Sample", marker_color="#2A9D8F", opacity=0.7))
    fig.add_vline(x=sample_mean, line_dash="solid", line_color="#E63946", line_width=2,
                  annotation_text=f"Sample Mean: {sample_mean:.2f}")
    fig.add_vline(x=pop_mean, line_dash="dash", line_color="#264653", line_width=2,
                  annotation_text=f"Population Mean: {pop_mean:.2f}")
    fig.add_vrect(x0=ci_lower, x1=ci_upper, fillcolor="#F4A261", opacity=0.2,
                  annotation_text=f"{conf_level}% CI", line_width=0)
    apply_common_layout(fig, title=f"{conf_level}% CI for Mean {FEATURE_LABELS[ci_feature]} in {ci_city}")
    st.plotly_chart(fig, use_container_width=True)

mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Sample Mean", f"{sample_mean:.2f}")
mcol2.metric("CI Lower", f"{ci_lower:.2f}")
mcol3.metric("CI Upper", f"{ci_upper:.2f}")
mcol4.metric("CI Width", f"{ci_upper - ci_lower:.2f}")

captures = ci_lower <= pop_mean <= ci_upper
if captures:
    st.success(f"This CI captures the true population mean ({pop_mean:.2f}).")
else:
    st.error(f"This CI does NOT capture the true population mean ({pop_mean:.2f}).")

st.divider()

# ---------------------------------------------------------------------------
# 3. 100 CIs Simulation
# ---------------------------------------------------------------------------
st.subheader("The Parade of 100 Confidence Intervals")

st.markdown(
    "Okay, so I claimed that a 95% CI means roughly 95 out of 100 intervals will contain "
    "the true mean. Let's actually check that. We'll draw **100 random samples**, compute "
    "a CI for each one, and count how many manage to lasso the true population mean. "
    "If the theory works, about 95 should succeed. If substantially fewer do, we should "
    "start questioning our assumptions."
)

sim_conf = st.slider("Confidence level for simulation (%)", 80, 99, 95, key="sim_conf")
sim_n = st.slider("Sample size per draw", 30, 2000, 200, step=10, key="sim_n")

z_sim = sp_stats.norm.ppf(1 - (1 - sim_conf / 100) / 2)

rng_sim = np.random.RandomState(123)
ci_results = []
for i in range(100):
    s = rng_sim.choice(pop_data, size=min(sim_n, len(pop_data)), replace=True)
    m = s.mean()
    se = s.std(ddof=1) / np.sqrt(len(s))
    lo = m - z_sim * se
    hi = m + z_sim * se
    captures_pop = lo <= pop_mean <= hi
    ci_results.append({"trial": i + 1, "mean": m, "lower": lo, "upper": hi, "captures": captures_pop})

ci_df = pd.DataFrame(ci_results)
n_captured = ci_df["captures"].sum()

fig_sim = go.Figure()
for _, row in ci_df.iterrows():
    color = "#2A9D8F" if row["captures"] else "#E63946"
    fig_sim.add_trace(go.Scatter(
        x=[row["lower"], row["upper"]], y=[row["trial"], row["trial"]],
        mode="lines", line=dict(color=color, width=1.5),
        showlegend=False,
    ))
    fig_sim.add_trace(go.Scatter(
        x=[row["mean"]], y=[row["trial"]],
        mode="markers", marker=dict(color=color, size=3),
        showlegend=False,
    ))

fig_sim.add_vline(x=pop_mean, line_dash="dash", line_color="#264653", line_width=2,
                  annotation_text=f"True Mean: {pop_mean:.2f}")
apply_common_layout(fig_sim, title=f"100 CIs at {sim_conf}% Level (n={sim_n})", height=600)
fig_sim.update_yaxes(title_text="Trial", range=[0, 101])
fig_sim.update_xaxes(title_text=FEATURE_LABELS[ci_feature])
st.plotly_chart(fig_sim, use_container_width=True)

cmet1, cmet2 = st.columns(2)
cmet1.metric("CIs capturing true mean", f"{n_captured}/100")
cmet2.metric("Expected", f"~{sim_conf}/100")

insight_box(
    f"Out of 100 intervals, {n_captured} captured the true mean. We expected about {sim_conf}. "
    f"The red intervals are the unlucky ones -- they missed. And here's the key insight: before "
    "we computed them, we had no way of knowing which ones would miss. Each individual interval "
    "either contains the truth or it doesn't. The {sim_conf}% is a property of the assembly line, "
    "not of any particular product coming off it."
)

st.divider()

# ---------------------------------------------------------------------------
# 4. Bootstrap Confidence Interval
# ---------------------------------------------------------------------------
st.subheader("The Bootstrap: When You Can't Assume Normality")

concept_box(
    "Bootstrap CI -- The 'Pull Yourself Up By Your Bootstraps' Method",
    "Here's a beautifully strange idea: what if we don't know the population distribution, "
    "but we have a sample? We can treat the sample <em>as if it were the population</em>, "
    "draw thousands of resamples from it (with replacement), compute our statistic each time, "
    "and use the resulting distribution to build a CI. This is the <b>bootstrap</b>, and it's "
    "one of the cleverest tricks in statistics. It's distribution-free, it works for any "
    "statistic (means, medians, ratios, whatever you like), and it relies on nothing more "
    "than the assumption that your sample is reasonably representative of reality.",
)

boot_city = st.selectbox("City for bootstrap", CITY_LIST, key="boot_city")
boot_feat = st.selectbox(
    "Feature for bootstrap",
    FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    key="boot_feat",
)
boot_conf = st.slider("Bootstrap CI level (%)", 80, 99, 95, key="boot_conf")
boot_nboot = st.slider("Number of bootstrap resamples", 500, 5000, 1000, step=500, key="boot_nboot")

boot_data = fdf[fdf["city"] == boot_city][boot_feat].dropna().values
boot_lower, boot_upper, boot_dist = bootstrap_ci(
    boot_data, stat_func=np.mean, n_boot=boot_nboot, ci=boot_conf, seed=42
)

fig_boot = go.Figure()
fig_boot.add_trace(go.Histogram(x=boot_dist, nbinsx=60, name="Bootstrap Means",
                                marker_color="#7209B7", opacity=0.7))
fig_boot.add_vline(x=boot_lower, line_dash="dash", line_color="#E63946", line_width=2,
                   annotation_text=f"Lower: {boot_lower:.2f}")
fig_boot.add_vline(x=boot_upper, line_dash="dash", line_color="#E63946", line_width=2,
                   annotation_text=f"Upper: {boot_upper:.2f}")
fig_boot.add_vline(x=np.mean(boot_data), line_dash="solid", line_color="#264653", line_width=2,
                   annotation_text=f"Sample Mean: {np.mean(boot_data):.2f}")
apply_common_layout(fig_boot, title=f"Bootstrap Distribution of Mean {FEATURE_LABELS[boot_feat]} ({boot_city})")
st.plotly_chart(fig_boot, use_container_width=True)

bm1, bm2, bm3 = st.columns(3)
bm1.metric("Bootstrap CI Lower", f"{boot_lower:.2f}")
bm2.metric("Bootstrap CI Upper", f"{boot_upper:.2f}")
bm3.metric("CI Width", f"{boot_upper - boot_lower:.2f}")

st.divider()

# ---------------------------------------------------------------------------
# 5. Code Example
# ---------------------------------------------------------------------------
code_example("""
import numpy as np
from scipy import stats

# Normal-approximation CI
sample = city_df['temperature_c'].sample(500, random_state=42)
mean = sample.mean()
se = sample.std(ddof=1) / np.sqrt(len(sample))
z = stats.norm.ppf(0.975)  # 95% CI
ci_lower, ci_upper = mean - z * se, mean + z * se
print(f"95% CI: ({ci_lower:.2f}, {ci_upper:.2f})")

# Bootstrap CI
def bootstrap_ci(data, n_boot=1000, ci=95):
    boot_means = [np.mean(np.random.choice(data, len(data), replace=True))
                  for _ in range(n_boot)]
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lower, upper

lo, hi = bootstrap_ci(sample.values)
print(f"Bootstrap 95% CI: ({lo:.2f}, {hi:.2f})")
""")

st.divider()

# ---------------------------------------------------------------------------
# 6. Quiz
# ---------------------------------------------------------------------------
quiz(
    "If you increase the confidence level from 90% to 99%, what happens to the CI width?",
    [
        "It gets narrower",
        "It gets wider",
        "It stays the same",
        "It depends on the sample size",
    ],
    correct_idx=1,
    explanation="You're demanding more certainty, and certainty isn't free. The interval has to widen to keep its coverage promise. Think of it like a net -- if you want to be more sure you'll catch the fish, you need a bigger net.",
    key="ch11_quiz1",
)

quiz(
    "A 95% CI of (18.2, 22.5) for mean temperature means:",
    [
        "There is a 95% probability the true mean is between 18.2 and 22.5",
        "95% of temperatures fall between 18.2 and 22.5",
        "If we repeated this procedure many times, 95% of intervals would contain the true mean",
        "The sample mean is 95% accurate",
    ],
    correct_idx=2,
    explanation="This is the correct frequentist interpretation, and it's famously unintuitive. The true mean is fixed -- it doesn't have a probability of being anywhere. The 95% describes the reliability of the procedure, not the location of the parameter.",
    key="ch11_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 7. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "A confidence interval is how honest statisticians admit they're not sure about their point estimate. It quantifies the uncertainty.",
    "Three knobs control CI width: sample size (more data = narrower), variability (more spread = wider), and confidence level (more confidence = wider). You can't optimize all three at once.",
    "The correct interpretation is about the procedure, not any single interval. 95% of intervals constructed this way will capture the true mean. Yours might be one of the unlucky 5%.",
    "The bootstrap is a remarkably clever workaround for when you can't assume normality -- just resample from your data and let the empirical distribution do the heavy lifting.",
    "If someone tells you they're '95% confident the true value is in this range,' they're technically wrong in frequentist statistics, but they've got the right spirit.",
])

navigation(
    prev_label="Ch 10: Advanced Visualization",
    prev_page="10_Advanced_Visualization.py",
    next_label="Ch 12: Hypothesis Testing",
    next_page="12_Hypothesis_Testing.py",
)
