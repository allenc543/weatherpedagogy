"""Chapter 15: Nonparametric Tests — Mann-Whitney, Kruskal-Wallis, KS test."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as sp_stats

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, color_map, histogram_chart, box_chart
from utils.stats_helpers import (
    normality_test, perform_ttest, ks_test, perform_anova,
)
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)
from utils.constants import CITY_LIST, FEATURE_COLS, FEATURE_LABELS, CITY_COLORS

# ---------------------------------------------------------------------------
df = load_data()
fdf = sidebar_filters(df)

chapter_header(15, "Nonparametric Tests", part="III")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "What to Do When Your Data Won't Behave Normally",
    "Everything we've done so far -- t-tests, ANOVA -- has a dirty little secret: it assumes "
    "your data is approximately normally distributed. For temperature? That's usually fine. But "
    "wind speed? Wind speed is heavily right-skewed (lots of calm days, occasional gusts). "
    "Humidity in arid cities? Also not normal. When the normality assumption falls apart, we need "
    "tests that don't care about the shape of your distribution. These are <b>nonparametric tests</b>, "
    "and their trick is elegant: instead of comparing raw values, they compare <b>ranks</b>. Is "
    "the value the 50th-largest or the 500th-largest? That's all they need to know.",
)

st.markdown("### The Translation Table: Parametric to Nonparametric")
comp_data = pd.DataFrame({
    "Scenario": ["2 groups", "3+ groups", "Distribution comparison"],
    "Parametric": ["Independent t-test", "One-way ANOVA", "--"],
    "Nonparametric": ["Mann-Whitney U", "Kruskal-Wallis H", "Kolmogorov-Smirnov"],
    "Tests Ranks Of": ["Central tendency", "Central tendency", "Full distribution shape"],
})
st.dataframe(comp_data, use_container_width=True, hide_index=True)

col1, col2 = st.columns(2)
with col1:
    formula_box(
        "Mann-Whitney U statistic",
        r"\underbrace{U}_{\text{test statistic}} = \underbrace{n_1}_{\text{group 1 size}} \underbrace{n_2}_{\text{group 2 size}} + \frac{n_1(n_1+1)}{2} - \underbrace{R_1}_{\text{rank sum, group 1}}",
        "where R_1 is the sum of ranks for group 1. The core question: if you randomly "
        "picked one observation from each group, would one group's values tend to be "
        "larger than the other's? U captures this tendency.",
    )
with col2:
    formula_box(
        "Kruskal-Wallis H statistic",
        r"\underbrace{H}_{\text{test statistic}} = \frac{12}{\underbrace{N}_{\text{total obs}}(N+1)} \sum_{i=1}^{\underbrace{k}_{\text{num groups}}} \frac{\underbrace{R_i^2}_{\text{rank sum squared}}}{\underbrace{n_i}_{\text{group size}}} - 3(N+1)",
        "The ANOVA of ranks. It asks: do the average ranks differ across k groups more "
        "than you'd expect by chance? If H is large, at least one group's values tend "
        "to be systematically higher or lower than the others.",
    )

st.divider()

# ---------------------------------------------------------------------------
# 2. Normality Check: Motivating Nonparametric Tests
# ---------------------------------------------------------------------------
st.subheader("First, Let's See How Non-Normal Our Data Actually Is")

st.markdown(
    "Before reaching for nonparametric tests, we should check whether we actually need them. "
    "The **Shapiro-Wilk test** tests the null hypothesis that data comes from a normal distribution. "
    "If p < 0.05, we reject normality. (Spoiler: with large samples, almost everything fails "
    "this test. The question isn't really 'is it perfectly normal?' but 'is it non-normal enough "
    "to worry about?')"
)

norm_results = []
for feat in FEATURE_COLS:
    for city in sorted(fdf["city"].unique()):
        data = fdf[fdf["city"] == city][feat].dropna().values
        if len(data) > 10:
            stat, p = normality_test(data)
            norm_results.append({
                "City": city,
                "Feature": FEATURE_LABELS[feat],
                "Shapiro W": round(stat, 4),
                "p-value": f"{p:.2e}",
                "Normal?": "Yes" if p >= 0.05 else "No",
            })

if norm_results:
    norm_df = pd.DataFrame(norm_results)
    st.dataframe(norm_df, use_container_width=True, hide_index=True)

    n_fail = (norm_df["Normal?"] == "No").sum()
    n_total = len(norm_df)
    insight_box(
        f"{n_fail} out of {n_total} city-feature combinations fail the normality test. "
        "In a sense, this is not surprising -- real-world data almost never perfectly follows "
        "a textbook bell curve. Wind speed in particular has that characteristic right-skewed "
        "shape (lots of gentle breezes, rare howling gales). This is exactly the kind of data "
        "where nonparametric tests earn their keep."
    )

# Show wind speed distribution as motivation
st.markdown("#### Exhibit A: Wind Speed Is Not Normal (and That's Okay)")
fig_wind = go.Figure()
for city in sorted(fdf["city"].unique()):
    w = fdf[fdf["city"] == city]["wind_speed_kmh"].dropna()
    if len(w) > 0:
        fig_wind.add_trace(go.Histogram(
            x=w, name=city, marker_color=CITY_COLORS.get(city, "#636EFA"),
            opacity=0.5, nbinsx=50,
        ))
fig_wind.update_layout(barmode="overlay")
apply_common_layout(fig_wind, title="Wind Speed: Clearly Non-Normal (Right-Skewed)")
fig_wind.update_xaxes(title_text="Wind Speed (km/h)")
st.plotly_chart(fig_wind, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 3. Interactive: Parametric vs Nonparametric Side by Side
# ---------------------------------------------------------------------------
st.subheader("The Head-to-Head: Parametric vs Nonparametric")

np_col1, np_col2 = st.columns([1, 2])

with np_col1:
    np_feat = st.selectbox(
        "Feature",
        FEATURE_COLS,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
        index=2,  # wind speed by default
        key="np_feat",
    )
    np_city1 = st.selectbox("Group 1 (City)", CITY_LIST, index=0, key="np_c1")
    np_city2 = st.selectbox("Group 2 (City)", CITY_LIST, index=4, key="np_c2")

g1_data = fdf[fdf["city"] == np_city1][np_feat].dropna()
g2_data = fdf[fdf["city"] == np_city2][np_feat].dropna()

if len(g1_data) > 0 and len(g2_data) > 0:
    with np_col2:
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Histogram(
            x=g1_data, name=np_city1,
            marker_color=CITY_COLORS.get(np_city1, "#636EFA"),
            opacity=0.6, nbinsx=50,
        ))
        fig_comp.add_trace(go.Histogram(
            x=g2_data, name=np_city2,
            marker_color=CITY_COLORS.get(np_city2, "#EF553B"),
            opacity=0.6, nbinsx=50,
        ))
        fig_comp.update_layout(barmode="overlay")
        apply_common_layout(fig_comp, title=f"{FEATURE_LABELS[np_feat]}: {np_city1} vs {np_city2}")
        st.plotly_chart(fig_comp, use_container_width=True)

    # Run both tests
    st.markdown("### The Verdict: Do They Agree?")
    param_col, nonparam_col = st.columns(2)

    # Parametric: t-test
    t_result = perform_ttest(g1_data, g2_data)
    with param_col:
        st.markdown("#### Parametric: Welch's t-test")
        st.metric("t-statistic", f"{t_result['t_stat']:.3f}")
        st.metric("p-value", f"{t_result['p_value']:.2e}")
        st.metric("Cohen's d", f"{t_result['cohens_d']:.3f}")
        if t_result["p_value"] < 0.05:
            st.success("Significant at alpha = 0.05")
        else:
            st.info("Not significant at alpha = 0.05")

    # Nonparametric: Mann-Whitney U
    u_stat, mw_p = sp_stats.mannwhitneyu(g1_data, g2_data, alternative="two-sided")
    # Rank-biserial correlation as effect size
    n1, n2 = len(g1_data), len(g2_data)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

    with nonparam_col:
        st.markdown("#### Nonparametric: Mann-Whitney U")
        st.metric("U-statistic", f"{u_stat:.0f}")
        st.metric("p-value", f"{mw_p:.2e}")
        st.metric("Rank-Biserial r", f"{rank_biserial:.3f}")
        if mw_p < 0.05:
            st.success("Significant at alpha = 0.05")
        else:
            st.info("Not significant at alpha = 0.05")

    # Agreement check
    both_sig = t_result["p_value"] < 0.05 and mw_p < 0.05
    both_ns = t_result["p_value"] >= 0.05 and mw_p >= 0.05

    if both_sig or both_ns:
        insight_box(
            "Both tests agree, which is reassuring. When the parametric and nonparametric "
            "approaches reach the same conclusion, you can be fairly confident the result is "
            "robust to distributional assumptions."
        )
    else:
        warning_box(
            "The tests disagree, and this is exactly the situation nonparametric tests were designed "
            "for. The t-test assumed something about the distribution that may not hold here. When "
            "they conflict, the nonparametric result is generally more trustworthy -- it made fewer "
            "assumptions, so there are fewer ways for it to go wrong."
        )

    # KS test
    st.divider()
    st.markdown("### The Kolmogorov-Smirnov Test: Comparing Entire Distributions")

    concept_box(
        "KS Test -- Beyond Just the Average",
        "The Mann-Whitney test asks 'does one group tend to have higher values?' But what if "
        "two groups have the same median but completely different shapes? One might be narrow "
        "and peaked while the other is flat and spread out. The <b>KS test</b> catches this "
        "because it compares the <b>entire cumulative distribution</b> -- shape, spread, location, "
        "everything. It finds the maximum vertical gap between the two CDFs. If that gap is too "
        "large to explain by chance, the distributions are different.",
    )

    ks_result = ks_test(g1_data.values, g2_data.values)

    ks1, ks2 = st.columns(2)
    ks1.metric("KS Statistic", f"{ks_result['ks_stat']:.4f}")
    ks2.metric("p-value", f"{ks_result['p_value']:.2e}")

    # ECDF plot
    fig_ecdf = go.Figure()
    for city, data, color in [(np_city1, g1_data, CITY_COLORS.get(np_city1, "#636EFA")),
                               (np_city2, g2_data, CITY_COLORS.get(np_city2, "#EF553B"))]:
        sorted_data = np.sort(data)
        ecdf_y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        fig_ecdf.add_trace(go.Scatter(
            x=sorted_data, y=ecdf_y, mode="lines", name=city,
            line=dict(color=color, width=2),
        ))
    apply_common_layout(fig_ecdf, title="Empirical CDF Comparison (KS Test)")
    fig_ecdf.update_xaxes(title_text=FEATURE_LABELS[np_feat])
    fig_ecdf.update_yaxes(title_text="Cumulative Probability")
    st.plotly_chart(fig_ecdf, use_container_width=True)

    if ks_result["p_value"] < 0.05:
        st.success(
            f"KS test is significant (D = {ks_result['ks_stat']:.4f}, p = {ks_result['p_value']:.2e}). "
            "The two distributions are significantly different."
        )
    else:
        st.info("KS test is not significant. No evidence of different distributions.")

else:
    st.warning("Not enough data for selected cities. Adjust sidebar filters.")

st.divider()

# ---------------------------------------------------------------------------
# 4. Kruskal-Wallis: Nonparametric ANOVA
# ---------------------------------------------------------------------------
st.subheader("Kruskal-Wallis: ANOVA Without the Normality Assumption")

kw_feat = st.selectbox(
    "Feature for Kruskal-Wallis",
    FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    index=2,
    key="kw_feat",
)

kw_cities = sorted(fdf["city"].unique())
kw_groups = []
kw_labels = []
for c in kw_cities:
    g = fdf[fdf["city"] == c][kw_feat].dropna().values
    if len(g) > 0:
        kw_groups.append(g)
        kw_labels.append(c)

if len(kw_groups) >= 3:
    # Parametric ANOVA
    anova_res = perform_anova(*kw_groups)

    # Kruskal-Wallis
    kw_stat, kw_p = sp_stats.kruskal(*kw_groups)

    # Epsilon-squared effect size: H / ((n^2 - 1) / (n + 1))
    n_total = sum(len(g) for g in kw_groups)
    epsilon_sq = (kw_stat - len(kw_groups) + 1) / (n_total - len(kw_groups))

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown("#### Parametric: One-Way ANOVA")
        st.metric("F-statistic", f"{anova_res['f_stat']:.2f}")
        st.metric("p-value", f"{anova_res['p_value']:.2e}")
    with comp_col2:
        st.markdown("#### Nonparametric: Kruskal-Wallis")
        st.metric("H-statistic", f"{kw_stat:.2f}")
        st.metric("p-value", f"{kw_p:.2e}")
        st.metric("Epsilon-squared", f"{epsilon_sq:.4f}")

st.divider()

# ---------------------------------------------------------------------------
# 5. Code Example
# ---------------------------------------------------------------------------
code_example("""
from scipy import stats

# Normality test
stat, p = stats.shapiro(city_df['wind_speed_kmh'].sample(5000))
print(f"Shapiro-Wilk: W={stat:.4f}, p={p:.2e}")

# Mann-Whitney U test (2 groups)
dallas_wind = df[df['city']=='Dallas']['wind_speed_kmh']
houston_wind = df[df['city']=='Houston']['wind_speed_kmh']
u_stat, p = stats.mannwhitneyu(dallas_wind, houston_wind, alternative='two-sided')
print(f"Mann-Whitney U={u_stat:.0f}, p={p:.2e}")

# Kruskal-Wallis (3+ groups)
groups = [df[df['city']==c]['wind_speed_kmh'].dropna() for c in df['city'].unique()]
h_stat, p = stats.kruskal(*groups)
print(f"Kruskal-Wallis H={h_stat:.2f}, p={p:.2e}")

# KS test (distribution comparison)
ks_stat, p = stats.ks_2samp(dallas_wind, houston_wind)
print(f"KS D={ks_stat:.4f}, p={p:.2e}")
""")

st.divider()

# ---------------------------------------------------------------------------
# 6. Quiz
# ---------------------------------------------------------------------------
quiz(
    "When should you prefer a nonparametric test over a parametric one?",
    [
        "Always, because they are more powerful",
        "When the sample size is very large",
        "When the data violate the normality assumption",
        "When you want a smaller p-value",
    ],
    correct_idx=2,
    explanation="Nonparametric tests shine when your data doesn't fit the normal distribution assumption. They're not always better -- when normality holds, parametric tests actually have more power. It's a tradeoff: nonparametric tests make fewer assumptions but pay a small power penalty for the privilege.",
    key="ch15_quiz1",
)

quiz(
    "The Mann-Whitney U test compares:",
    [
        "The means of two groups",
        "The variances of two groups",
        "The ranks of observations between two groups",
        "The correlation between two variables",
    ],
    correct_idx=2,
    explanation="Mann-Whitney converts everything to ranks and asks: does one group tend to have higher ranks than the other? It doesn't care about the actual values, just the ordering. This is what makes it robust to outliers and non-normality.",
    key="ch15_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 7. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Nonparametric tests work on ranks instead of raw values, which means they don't care whether your data looks like a bell curve, a ski jump, or a camel's back.",
    "Mann-Whitney U is the nonparametric t-test (2 groups); Kruskal-Wallis H is the nonparametric ANOVA (3+ groups). Same questions, fewer assumptions.",
    "The KS test is the most comprehensive: it compares entire distribution shapes, not just averages. Two groups can have the same mean but fail the KS test because one is spread wide and the other is tightly clustered.",
    "Wind speed in our dataset is a textbook example of non-normal data -- that right-skewed shape is exactly why nonparametric methods exist.",
    "There's a real tradeoff: when normality holds, parametric tests are slightly more powerful. But when it doesn't, nonparametric tests are more trustworthy. Since real-world data is rarely perfectly normal, nonparametric tests are an important tool in the kit.",
])

navigation(
    prev_label="Ch 14: ANOVA",
    prev_page="14_ANOVA.py",
    next_label="Ch 16: Correlation Analysis",
    next_page="16_Correlation_Analysis.py",
)
