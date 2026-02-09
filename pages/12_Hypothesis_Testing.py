"""Chapter 12: Hypothesis Testing — t-tests, p-values, Type I/II errors, effect size."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as sp_stats

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, color_map, histogram_chart, box_chart
from utils.stats_helpers import perform_ttest, cohens_d, descriptive_stats
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)
from utils.constants import CITY_LIST, FEATURE_COLS, FEATURE_LABELS, CITY_COLORS

# ---------------------------------------------------------------------------
df = load_data()
fdf = sidebar_filters(df)

chapter_header(12, "Hypothesis Testing", part="III")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "Hypothesis Testing: A Framework for Structured Arguing With Data",
    "Here's the basic setup. You have a question -- say, 'Is Dallas hotter than Houston?' -- "
    "and you want to answer it with data rather than vibes. So you start by assuming the boring "
    "answer is true: <b>there's no difference</b> (this is your null hypothesis, H0). Then you "
    "collect data and ask: 'If there really were no difference, how surprised would I be to see "
    "data this extreme?' If the answer is 'very surprised,' you reject the boring answer. If "
    "the answer is 'meh, this could easily happen by chance,' you shrug and say you can't rule "
    "it out. That's the whole framework. Everything else is details.",
)

col1, col2 = st.columns(2)
with col1:
    formula_box(
        "t-test statistic (Welch's)",
        r"t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}",
        "The numerator is the difference you care about. The denominator is how much noise you'd expect. A big t means the signal is large relative to the noise.",
    )
with col2:
    formula_box(
        "Cohen's d (effect size)",
        r"d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}",
        "This tells you how big the difference actually is in standard-deviation units. "
        "A p-value tells you 'is the difference real?' while Cohen's d tells you 'is the difference "
        "worth caring about?' |d| < 0.2 = small, 0.5 = medium, 0.8 = large.",
    )

st.markdown("### The Two Ways You Can Be Wrong")
err_col1, err_col2 = st.columns(2)
with err_col1:
    st.markdown(
        "**Type I Error (The False Alarm):** You reject H0 when it's actually true. You declare "
        "'Dallas IS hotter!' when it really isn't. This is controlled by alpha, which is usually "
        "set at 0.05 -- meaning you're accepting a 5% false alarm rate. Think of it as the boy "
        "who cried wolf."
    )
with err_col2:
    st.markdown(
        "**Type II Error (The Missed Signal):** You fail to reject H0 when there really IS a "
        "difference. Dallas genuinely is hotter, but your data wasn't convincing enough to prove it. "
        "This is related to statistical power (1 - beta). Think of it as the wolf sneaking past "
        "while the boy is asleep."
    )

warning_box(
    "This is probably the single most misunderstood thing in all of statistics: a p-value is NOT "
    "the probability that H0 is true. It's the probability of seeing data this extreme <em>if</em> "
    "H0 were true. These sound similar but are completely different things. 'The probability of "
    "being wet given that it's raining' is not the same as 'the probability that it's raining "
    "given that you're wet.' (You might have just fallen into a pool.)"
)

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Compare Two Cities
# ---------------------------------------------------------------------------
st.subheader("Put It to the Test: Compare Two Cities")

col_left, col_right = st.columns([1, 2])

with col_left:
    city1 = st.selectbox("City 1", CITY_LIST, index=0, key="ht_city1")
    city2 = st.selectbox("City 2", CITY_LIST, index=4, key="ht_city2")
    feature = st.selectbox(
        "Feature to compare",
        FEATURE_COLS,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
        key="ht_feat",
    )
    alpha = st.select_slider("Significance level (alpha)", options=[0.01, 0.05, 0.10], value=0.05, key="ht_alpha")

    st.markdown("---")
    st.markdown("**Try these preset comparisons:**")
    if st.button("Dallas vs Houston (nearby, small effect)", key="ht_preset1"):
        st.session_state.ht_city1_idx = CITY_LIST.index("Dallas")
        st.session_state.ht_city2_idx = CITY_LIST.index("Houston")
        st.rerun()
    if st.button("NYC vs Los Angeles (large effect)", key="ht_preset2"):
        st.session_state.ht_city1_idx = CITY_LIST.index("NYC")
        st.session_state.ht_city2_idx = CITY_LIST.index("Los Angeles")
        st.rerun()

g1 = fdf[fdf["city"] == city1][feature].dropna()
g2 = fdf[fdf["city"] == city2][feature].dropna()

if len(g1) > 0 and len(g2) > 0:
    result = perform_ttest(g1, g2, equal_var=False)

    with col_right:
        # Overlapping histograms
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=g1, name=city1, marker_color=CITY_COLORS.get(city1, "#636EFA"),
            opacity=0.6, nbinsx=60,
        ))
        fig.add_trace(go.Histogram(
            x=g2, name=city2, marker_color=CITY_COLORS.get(city2, "#EF553B"),
            opacity=0.6, nbinsx=60,
        ))
        fig.update_layout(barmode="overlay")
        apply_common_layout(fig, title=f"{FEATURE_LABELS[feature]}: {city1} vs {city2}")
        fig.update_xaxes(title_text=FEATURE_LABELS[feature])
        fig.update_yaxes(title_text="Count")
        st.plotly_chart(fig, use_container_width=True)

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("t-statistic", f"{result['t_stat']:.3f}")
    m2.metric("p-value", f"{result['p_value']:.2e}")
    m3.metric("Cohen's d", f"{result['cohens_d']:.3f}")

    abs_d = abs(result["cohens_d"])
    if abs_d < 0.2:
        effect_label = "Negligible"
    elif abs_d < 0.5:
        effect_label = "Small"
    elif abs_d < 0.8:
        effect_label = "Medium"
    else:
        effect_label = "Large"
    m4.metric("Effect Size", effect_label)

    # Decision
    if result["p_value"] < alpha:
        st.success(
            f"**Reject H0** at alpha = {alpha}. The difference in mean {FEATURE_LABELS[feature]} "
            f"between {city1} ({g1.mean():.2f}) and {city2} ({g2.mean():.2f}) is statistically significant "
            f"(p = {result['p_value']:.2e})."
        )
    else:
        st.info(
            f"**Fail to reject H0** at alpha = {alpha}. No statistically significant difference "
            f"detected (p = {result['p_value']:.2e})."
        )

    # Descriptive stats table
    st.markdown("#### The Numbers Behind the Test")
    stats1 = descriptive_stats(g1)
    stats2 = descriptive_stats(g2)
    comp_df = pd.DataFrame({city1: stats1, city2: stats2}).round(3)
    st.dataframe(comp_df, use_container_width=True)

    st.divider()

    # ---------------------------------------------------------------------------
    # 3. Effect Size Visualization
    # ---------------------------------------------------------------------------
    st.subheader("Why P-Values Aren't Enough: The Effect Size Story")

    st.markdown(
        "Here's the dirty secret of hypothesis testing with big data: **statistical significance "
        "is absurdly easy to achieve when you have enough observations.** With 17,000+ data points "
        "per city, even a 0.1-degree temperature difference becomes 'statistically significant.' "
        "But does anyone actually care about a tenth of a degree? This is where effect size comes "
        "in -- it tells you whether the difference is large enough to matter in the real world."
    )

    # Generate normal curves for the two groups
    x_range = np.linspace(
        min(g1.mean() - 4 * g1.std(), g2.mean() - 4 * g2.std()),
        max(g1.mean() + 4 * g1.std(), g2.mean() + 4 * g2.std()),
        500,
    )
    y1 = sp_stats.norm.pdf(x_range, g1.mean(), g1.std())
    y2 = sp_stats.norm.pdf(x_range, g2.mean(), g2.std())

    fig_eff = go.Figure()
    fig_eff.add_trace(go.Scatter(
        x=x_range, y=y1, mode="lines", name=city1,
        line=dict(color=CITY_COLORS.get(city1, "#636EFA"), width=2),
        fill="tozeroy", opacity=0.4,
    ))
    fig_eff.add_trace(go.Scatter(
        x=x_range, y=y2, mode="lines", name=city2,
        line=dict(color=CITY_COLORS.get(city2, "#EF553B"), width=2),
        fill="tozeroy", opacity=0.4,
    ))
    fig_eff.add_vline(x=g1.mean(), line_dash="dash",
                      line_color=CITY_COLORS.get(city1, "#636EFA"), line_width=1)
    fig_eff.add_vline(x=g2.mean(), line_dash="dash",
                      line_color=CITY_COLORS.get(city2, "#EF553B"), line_width=1)
    apply_common_layout(fig_eff, title=f"Effect Size: Cohen's d = {result['cohens_d']:.3f} ({effect_label})")
    fig_eff.update_xaxes(title_text=FEATURE_LABELS[feature])
    fig_eff.update_yaxes(title_text="Density")
    st.plotly_chart(fig_eff, use_container_width=True)

    insight_box(
        f"With n={len(g1):,} and n={len(g2):,} observations, even a Cohen's d of {abs_d:.3f} "
        f"({'basically nothing' if abs_d < 0.2 else 'a moderate difference' if abs_d < 0.5 else 'a substantial gap'}) "
        f"gets flagged as statistically significant. The p-value tells you the signal is real; "
        f"Cohen's d tells you whether the signal is worth writing home about. Always report both."
    )
else:
    st.warning("Not enough data for the selected cities. Adjust filters in the sidebar.")

st.divider()

# ---------------------------------------------------------------------------
# 4. All Pairwise Comparisons
# ---------------------------------------------------------------------------
st.subheader("The Full Bracket: Every City vs Every City")

pw_feat = st.selectbox(
    "Feature for pairwise comparison",
    FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    key="ht_pw_feat",
)

cities_avail = sorted(fdf["city"].unique())
pairs = []
for i, c1 in enumerate(cities_avail):
    for c2 in cities_avail[i + 1:]:
        d1 = fdf[fdf["city"] == c1][pw_feat].dropna()
        d2 = fdf[fdf["city"] == c2][pw_feat].dropna()
        if len(d1) > 0 and len(d2) > 0:
            res = perform_ttest(d1, d2)
            pairs.append({
                "City 1": c1,
                "City 2": c2,
                "Mean 1": round(d1.mean(), 2),
                "Mean 2": round(d2.mean(), 2),
                "Diff": round(d1.mean() - d2.mean(), 2),
                "t-stat": round(res["t_stat"], 3),
                "p-value": f"{res['p_value']:.2e}",
                "Cohen's d": round(res["cohens_d"], 3),
                "Significant (0.05)": "Yes" if res["p_value"] < 0.05 else "No",
            })

if pairs:
    st.dataframe(pd.DataFrame(pairs), use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# 5. Code Example
# ---------------------------------------------------------------------------
code_example("""
from scipy import stats
import numpy as np

dallas = df[df['city'] == 'Dallas']['temperature_c']
houston = df[df['city'] == 'Houston']['temperature_c']

# Welch's t-test (unequal variances)
t_stat, p_value = stats.ttest_ind(dallas, houston, equal_var=False)
print(f"t = {t_stat:.3f}, p = {p_value:.2e}")

# Cohen's d effect size
pooled_std = np.sqrt(((len(dallas)-1)*dallas.var() + (len(houston)-1)*houston.var())
                     / (len(dallas) + len(houston) - 2))
d = (dallas.mean() - houston.mean()) / pooled_std
print(f"Cohen's d = {d:.3f}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Reject H0: means are significantly different")
else:
    print("Fail to reject H0: no significant difference detected")
""")

st.divider()

# ---------------------------------------------------------------------------
# 6. Quiz
# ---------------------------------------------------------------------------
quiz(
    "A p-value of 0.03 means:",
    [
        "There is a 3% chance H0 is true",
        "There is a 3% chance of observing data this extreme if H0 were true",
        "The effect size is 0.03",
        "We are 97% confident H1 is true",
    ],
    correct_idx=1,
    explanation="The p-value answers: 'How weird is my data under the null hypothesis?' A p-value of 0.03 means: if there really were no difference, you'd see data this extreme only 3% of the time. It says nothing about the probability that H0 is true -- that requires Bayes' theorem and a prior.",
    key="ch12_quiz1",
)

quiz(
    "With n = 50,000 per group, you find p < 0.001 but Cohen's d = 0.02. What should you conclude?",
    [
        "The effect is both significant and meaningful",
        "The test is broken because d should be larger",
        "Statistically significant but practically negligible effect",
        "You should increase the sample size",
    ],
    correct_idx=2,
    explanation="This is the classic big-data trap. With enough observations, you can detect differences so tiny they'd need a microscope to see. The p-value says 'yes, the difference is real,' but Cohen's d says 'the difference is 0.02 standard deviations -- basically nothing.' Always ask both questions.",
    key="ch12_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 7. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Hypothesis testing gives you a structured way to ask 'is this difference real or just noise?' You assume no difference (H0), check how surprising your data is, and decide accordingly.",
    "The p-value measures how surprised you should be under H0 -- not the probability H0 is true. Getting this wrong is practically a rite of passage in statistics.",
    "Type I error is the false alarm (you thought the wolf was there); Type II is the missed detection (the wolf snuck past you). Alpha controls the first; power controls the second.",
    "Always report effect size alongside p-values. A 'significant' p-value with a tiny effect size is like technically winning an argument nobody cares about.",
    "With large samples, everything becomes significant. Practical significance -- the 'so what?' question -- is what actually matters for decisions.",
])

navigation(
    prev_label="Ch 11: Confidence Intervals",
    prev_page="11_Confidence_Intervals.py",
    next_label="Ch 13: A/B Testing",
    next_page="13_AB_Testing.py",
)
