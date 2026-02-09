"""Chapter 13: A/B Testing — Power analysis, multiple testing correction."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as sp_stats

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, color_map, box_chart, histogram_chart
from utils.stats_helpers import perform_ttest, cohens_d, descriptive_stats
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)
from utils.constants import (
    CITY_LIST, FEATURE_COLS, FEATURE_LABELS, CITY_COLORS,
    COASTAL_CITIES, INLAND_CITIES,
)

# ---------------------------------------------------------------------------
df = load_data()
fdf = sidebar_filters(df)

chapter_header(13, "A/B Testing", part="III")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "A/B Testing: The Gold Standard of 'Does This Actually Work?'",
    "A/B testing is what happens when you take hypothesis testing and hand it to a product team. "
    "You split users into two groups: Group A gets the status quo, Group B gets the shiny new "
    "thing, and you measure whether the shiny new thing actually makes a difference. In Silicon "
    "Valley, this means testing button colors and notification frequencies. In our weather context, "
    "we'll do something more interesting: we treat <b>Coastal cities (Houston, LA)</b> as Group A "
    "and <b>Inland cities (Dallas, Austin, San Antonio)</b> as Group B, and ask whether geography "
    "causes measurable differences in weather patterns. (Nature ran the A/B test for us; we just "
    "need to read the results.)",
)

st.markdown("### The Concepts That Make or Break an A/B Test")
col1, col2, col3 = st.columns(3)
with col1:
    concept_box("Statistical Power",
                "Power is your test's ability to detect a real effect when one exists. "
                "It's 1 minus the probability of a Type II error. The standard target "
                "is 80%, which sounds low until you realize that means you'll catch "
                "a real effect 4 out of 5 times. Most experiments that 'failed' actually "
                "just didn't have enough power.")
with col2:
    concept_box("The Multiple Testing Problem",
                "Run 1 test at alpha=0.05? You have a 5% false positive rate. "
                "Run 10 tests? Your chance of at least one false positive jumps to ~40%. "
                "Run 20? It's 64%. This is the statistical equivalent of buying 20 "
                "lottery tickets and being surprised when one wins. The more hypotheses "
                "you test, the more likely random noise looks like a real signal.")
with col3:
    concept_box("Bonferroni Correction",
                "The simplest fix for multiple testing: divide your alpha by the number "
                "of tests. Testing 10 hypotheses? Your new threshold is 0.005 instead of "
                "0.05. It's conservative -- possibly too conservative -- but it's dead simple "
                "and guarantees your overall false positive rate stays below alpha.")

formula_box(
    "Bonferroni Corrected Significance Level",
    r"\underbrace{\alpha_{\text{adjusted}}}_{\text{corrected threshold}} = \frac{\underbrace{\alpha}_{\text{original threshold}}}{\underbrace{m}_{\text{number of tests}}}",
    "where m is the number of comparisons. For 10 tests at alpha = 0.05: alpha_adj = 0.005. "
    "Harsh? Yes. But at least you won't publish embarrassing false positives.",
)

warning_box(
    "Here's a scenario that happens disturbingly often: a team runs 20 A/B tests, finds that "
    "one has p < 0.05, and writes a blog post about how they 'discovered' something. But with "
    "20 tests at alpha = 0.05, you'd EXPECT one false positive even if every null hypothesis "
    "is true. That's not a discovery; that's arithmetic."
)

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Coastal vs Inland
# ---------------------------------------------------------------------------
st.subheader("The Natural Experiment: Coastal vs Inland Cities")

st.markdown(
    f"**Group A (Coastal):** {', '.join(COASTAL_CITIES)}  \n"
    f"**Group B (Inland):** {', '.join(INLAND_CITIES)}"
)

ab_feat = st.selectbox(
    "Feature to test",
    FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    key="ab_feat",
)

coastal_data = fdf[fdf["city"].isin(COASTAL_CITIES)][ab_feat].dropna()
inland_data = fdf[fdf["city"].isin(INLAND_CITIES)][ab_feat].dropna()

if len(coastal_data) > 0 and len(inland_data) > 0:
    col_box, col_hist = st.columns(2)

    with col_box:
        fdf_ab = fdf[fdf["city"].isin(COASTAL_CITIES + INLAND_CITIES)].copy()
        fdf_ab["group"] = fdf_ab["city"].apply(
            lambda c: "Coastal" if c in COASTAL_CITIES else "Inland"
        )
        fig_box = go.Figure()
        for grp, color in [("Coastal", "#2A9D8F"), ("Inland", "#E63946")]:
            grp_data = fdf_ab[fdf_ab["group"] == grp][ab_feat]
            fig_box.add_trace(go.Box(y=grp_data, name=grp, marker_color=color))
        apply_common_layout(fig_box, title=f"{FEATURE_LABELS[ab_feat]}: Coastal vs Inland")
        st.plotly_chart(fig_box, use_container_width=True)

    with col_hist:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=coastal_data, name="Coastal", marker_color="#2A9D8F", opacity=0.6, nbinsx=60,
        ))
        fig_hist.add_trace(go.Histogram(
            x=inland_data, name="Inland", marker_color="#E63946", opacity=0.6, nbinsx=60,
        ))
        fig_hist.update_layout(barmode="overlay")
        apply_common_layout(fig_hist, title=f"Distribution: Coastal vs Inland")
        st.plotly_chart(fig_hist, use_container_width=True)

    result = perform_ttest(coastal_data, inland_data)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Coastal Mean", f"{coastal_data.mean():.2f}")
    m2.metric("Inland Mean", f"{inland_data.mean():.2f}")
    m3.metric("p-value", f"{result['p_value']:.2e}")
    m4.metric("Cohen's d", f"{result['cohens_d']:.3f}")

    if result["p_value"] < 0.05:
        st.success(
            f"The A/B test is significant at alpha=0.05. Coastal and inland cities differ "
            f"in {FEATURE_LABELS[ab_feat]} (Cohen's d = {result['cohens_d']:.3f})."
        )
    else:
        st.info("No significant difference detected between coastal and inland groups.")

st.divider()

# ---------------------------------------------------------------------------
# 3. Multiple Testing Correction
# ---------------------------------------------------------------------------
st.subheader("Multiple Testing: Why You Can't Just Test Everything and See What Sticks")

st.markdown(
    "We have 4 weather features we could compare between coastal and inland cities. If we "
    "test all 4 without adjusting our threshold, we're implicitly accepting a much higher "
    "false positive rate than we think. Let's see what happens when we apply the correction."
)

alpha_raw = st.select_slider(
    "Raw significance level (alpha)",
    options=[0.01, 0.05, 0.10],
    value=0.05,
    key="ab_alpha",
)

results_list = []
for feat in FEATURE_COLS:
    c_data = fdf[fdf["city"].isin(COASTAL_CITIES)][feat].dropna()
    i_data = fdf[fdf["city"].isin(INLAND_CITIES)][feat].dropna()
    if len(c_data) > 0 and len(i_data) > 0:
        res = perform_ttest(c_data, i_data)
        results_list.append({
            "Feature": FEATURE_LABELS[feat],
            "Coastal Mean": round(c_data.mean(), 2),
            "Inland Mean": round(i_data.mean(), 2),
            "t-stat": round(res["t_stat"], 3),
            "p-value": res["p_value"],
            "Cohen's d": round(res["cohens_d"], 3),
        })

if results_list:
    m = len(results_list)
    alpha_bonf = alpha_raw / m

    mc_df = pd.DataFrame(results_list)
    mc_df["Significant (raw)"] = mc_df["p-value"] < alpha_raw
    mc_df["Significant (Bonferroni)"] = mc_df["p-value"] < alpha_bonf
    mc_df["p-value"] = mc_df["p-value"].apply(lambda p: f"{p:.2e}")

    st.markdown(f"**Number of tests (m):** {m}")
    st.markdown(f"**Raw alpha:** {alpha_raw}")
    st.markdown(f"**Bonferroni-corrected alpha:** {alpha_bonf:.4f}")

    st.dataframe(mc_df, use_container_width=True, hide_index=True)

    raw_sig = mc_df["Significant (raw)"].sum()
    bonf_sig = mc_df["Significant (Bonferroni)"].sum()

    if raw_sig != bonf_sig:
        insight_box(
            f"And there it is. With the raw alpha, {raw_sig} features appear significant. After "
            f"Bonferroni correction, only {bonf_sig} survive. The ones that got knocked out were "
            "borderline cases where we can't be confident they aren't just noise masquerading as signal."
        )
    else:
        insight_box(
            f"All {raw_sig} significant results survive Bonferroni correction. When results are "
            f"this robust, the correction barely matters -- the effects are too strong for multiple "
            f"testing to explain away."
        )

st.divider()

# ---------------------------------------------------------------------------
# 4. Power Analysis & Sample Size Calculator
# ---------------------------------------------------------------------------
st.subheader("Power Analysis: How Much Data Do You Actually Need?")

concept_box(
    "Why You Should Do Power Analysis Before Your Experiment, Not After",
    "Imagine spending six months and $500,000 on an A/B test, only to realize afterward that "
    "you never had enough users to detect the effect you cared about. That's what happens when "
    "you skip power analysis. It answers the question: 'Given the effect size I want to detect "
    "and my tolerance for error, how many observations do I need?' Do this calculation before "
    "you start, or risk wasting everyone's time with an under-powered study.",
)

formula_box(
    "Sample Size for Two-Sample t-test",
    r"\underbrace{n}_{\text{required sample size}} = \left(\frac{\underbrace{z_{\alpha/2}}_{\text{critical value}} + \underbrace{z_{\beta}}_{\text{power quantile}}}{\underbrace{\delta}_{\text{min detectable diff}} / \underbrace{\sigma}_{\text{std dev}}}\right)^2",
    "where delta is the minimum detectable difference, sigma is the standard deviation, "
    "z_alpha/2 is the critical value, and z_beta relates to desired power.",
)

pwr_col1, pwr_col2 = st.columns([1, 2])

with pwr_col1:
    desired_power = st.slider("Desired power", 0.50, 0.99, 0.80, step=0.05, key="pwr_power")
    pwr_alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, step=0.01, key="pwr_alpha")
    effect_size_input = st.slider(
        "Expected effect size (Cohen's d)", 0.05, 2.0, 0.5, step=0.05, key="pwr_d"
    )

# Sample size calculation using the formula
z_alpha2 = sp_stats.norm.ppf(1 - pwr_alpha / 2)
z_beta = sp_stats.norm.ppf(desired_power)
n_required = int(np.ceil(((z_alpha2 + z_beta) / effect_size_input) ** 2))

with pwr_col2:
    st.metric("Required sample size (per group)", f"{n_required:,}")

    # Power curve
    n_range = np.arange(10, max(n_required * 3, 500), 5)
    power_vals = []
    for n_i in n_range:
        # Non-centrality parameter
        ncp = effect_size_input * np.sqrt(n_i / 2)
        crit = sp_stats.norm.ppf(1 - pwr_alpha / 2)
        pwr = 1 - sp_stats.norm.cdf(crit - ncp) + sp_stats.norm.cdf(-crit - ncp)
        power_vals.append(pwr)

    fig_pwr = go.Figure()
    fig_pwr.add_trace(go.Scatter(
        x=n_range, y=power_vals, mode="lines",
        line=dict(color="#2A9D8F", width=2), name="Power",
    ))
    fig_pwr.add_hline(y=desired_power, line_dash="dash", line_color="#E63946",
                      annotation_text=f"Target Power: {desired_power}")
    fig_pwr.add_vline(x=n_required, line_dash="dash", line_color="#F4A261",
                      annotation_text=f"n = {n_required}")
    apply_common_layout(fig_pwr, title="Power Curve: Sample Size vs Statistical Power")
    fig_pwr.update_xaxes(title_text="Sample Size (per group)")
    fig_pwr.update_yaxes(title_text="Power", range=[0, 1.05])
    st.plotly_chart(fig_pwr, use_container_width=True)

# Table of sample sizes for common effect sizes
st.markdown("#### How Many Subjects Do You Need? A Reference Table")
ref_data = []
for d in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]:
    n_ref = int(np.ceil(((z_alpha2 + z_beta) / d) ** 2))
    ref_data.append({"Cohen's d": d, "Effect": "Small" if d <= 0.2 else "Medium" if d <= 0.5 else "Large",
                     f"n per group (power={desired_power})": n_ref})
st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# 5. Code Example
# ---------------------------------------------------------------------------
code_example("""
from scipy import stats
import numpy as np

# A/B test: Coastal vs Inland humidity
coastal = df[df['city'].isin(['Houston', 'Los Angeles'])]['relative_humidity_pct']
inland = df[df['city'].isin(['Dallas', 'Austin', 'San Antonio'])]['relative_humidity_pct']

t_stat, p_value = stats.ttest_ind(coastal, inland, equal_var=False)
print(f"t = {t_stat:.3f}, p = {p_value:.2e}")

# Bonferroni correction for 4 tests
alpha = 0.05
m = 4  # number of features tested
alpha_bonf = alpha / m
print(f"Bonferroni alpha: {alpha_bonf:.4f}")
print(f"Significant after correction: {p_value < alpha_bonf}")

# Sample size calculation
def sample_size_ttest(effect_size, alpha=0.05, power=0.80):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return int(np.ceil(((z_alpha + z_beta) / effect_size) ** 2))

n = sample_size_ttest(0.5)
print(f"Required n per group for d=0.5: {n}")
""")

st.divider()

# ---------------------------------------------------------------------------
# 6. Quiz
# ---------------------------------------------------------------------------
quiz(
    "You run 20 independent hypothesis tests at alpha = 0.05. If all null hypotheses are true, "
    "how many false positives do you expect?",
    [
        "0",
        "1",
        "5",
        "20",
    ],
    correct_idx=1,
    explanation="Expected false positives = m * alpha = 20 * 0.05 = 1. One out of twenty will cross the threshold just by chance. This is why the multiple testing problem matters -- if you run enough tests, you're guaranteed to 'find' something, even in pure noise.",
    key="ch13_quiz1",
)

quiz(
    "What does the Bonferroni correction do?",
    [
        "Increases sample size",
        "Divides the significance level by the number of tests",
        "Adjusts the effect size",
        "Removes outliers before testing",
    ],
    correct_idx=1,
    explanation="Bonferroni divides alpha by the number of tests (m). It's the 'I'm going to be extra skeptical to compensate for all the opportunities I gave myself to find something' correction. Simple, effective, occasionally too paranoid.",
    key="ch13_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 7. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "A/B testing is hypothesis testing applied to real decisions. Two groups, one treatment, one question: did it make a difference?",
    "Statistical power is your probability of detecting a real effect. Most people target 80%, and most failed experiments failed because they never had enough power to begin with.",
    "The multiple testing problem is insidious: test enough hypotheses and noise will look like signal. Bonferroni correction is the simplest antidote, dividing your significance threshold by the number of tests.",
    "Power analysis is something you do BEFORE the experiment, not after. It tells you how much data you need. Skipping this step is like starting a road trip without checking if you have enough gas.",
    "Larger effect sizes need fewer samples to detect; tiny effects require enormous datasets. This is why detecting a 10-degree temperature difference takes 20 observations, but detecting a 0.1-degree difference takes 20,000.",
])

navigation(
    prev_label="Ch 12: Hypothesis Testing",
    prev_page="12_Hypothesis_Testing.py",
    next_label="Ch 14: ANOVA",
    next_page="14_ANOVA.py",
)
