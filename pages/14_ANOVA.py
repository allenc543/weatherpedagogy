"""Chapter 14: ANOVA — F-test, Tukey HSD, two-way ANOVA."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as sp_stats
from itertools import combinations

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, color_map, box_chart, heatmap_chart
from utils.stats_helpers import perform_anova, perform_ttest, cohens_d
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)
from utils.constants import (
    CITY_LIST, FEATURE_COLS, FEATURE_LABELS, CITY_COLORS, SEASON_ORDER,
)

# ---------------------------------------------------------------------------
df = load_data()
fdf = sidebar_filters(df)

chapter_header(14, "ANOVA (Analysis of Variance)", part="III")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "ANOVA: When Two Groups Aren't Enough",
    "You might be thinking: we already know how to compare two groups with a t-test. Why do we "
    "need ANOVA? Here's why. Suppose you want to know whether temperature differs across 6 "
    "cities. You could run 15 pairwise t-tests (6 choose 2), but then you'd have a massive "
    "multiple testing problem -- at alpha = 0.05, you'd expect almost one false positive just "
    "by chance. ANOVA solves this elegantly with a single test that asks: <b>'Are ANY of these "
    "group means different from each other?'</b> If the answer is yes, THEN you dig into which "
    "specific pairs differ using post-hoc tests. It's the statistical equivalent of 'first check "
    "if there's a fire, then figure out which room it's in.'",
)

col1, col2 = st.columns(2)
with col1:
    formula_box(
        "F-statistic",
        r"F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}} = "
        r"\frac{\text{Variance between groups}}{\text{Variance within groups}}",
        "The intuition is beautiful: if all groups have the same mean, then the variance "
        "between group means should be about the same as the variance within groups (both are "
        "just sampling noise). A large F means the between-group differences are too big to be "
        "explained by within-group variability alone.",
    )
with col2:
    formula_box(
        "Eta-squared (effect size for ANOVA)",
        r"\eta^2 = \frac{SS_{\text{between}}}{SS_{\text{total}}}",
        "What fraction of the total variability is explained by group membership? "
        "0.01 = small (group explains 1% of variance), 0.06 = medium, 0.14 = large. "
        "Think of it as ANOVA's version of R-squared.",
    )

st.markdown("### One-Way vs Two-Way ANOVA")
ow_col, tw_col = st.columns(2)
with ow_col:
    st.markdown(
        "**One-Way ANOVA:** One factor, multiple groups. 'Does mean temperature differ across "
        "our 6 cities?' This is the simplest version -- you have one categorical variable (city) "
        "and one continuous outcome (temperature)."
    )
with tw_col:
    st.markdown(
        "**Two-Way ANOVA:** Two factors and their interaction. 'Does the effect of city on "
        "temperature depend on the season?' Maybe Dallas-Houston differences are huge in summer "
        "but negligible in winter. The interaction term catches this kind of 'it depends' pattern."
    )

st.divider()

# ---------------------------------------------------------------------------
# 2. One-Way ANOVA
# ---------------------------------------------------------------------------
st.subheader("One-Way ANOVA: Do Cities Actually Differ?")

anova_feat = st.selectbox(
    "Select feature",
    FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    key="anova_feat",
)

cities_present = sorted(fdf["city"].unique())
groups = [fdf[fdf["city"] == c][anova_feat].dropna().values for c in cities_present]
groups = [g for g in groups if len(g) > 0]

if len(groups) >= 2:
    # Box plot
    fig_box = go.Figure()
    for city in cities_present:
        city_vals = fdf[fdf["city"] == city][anova_feat].dropna()
        if len(city_vals) > 0:
            fig_box.add_trace(go.Box(
                y=city_vals, name=city,
                marker_color=CITY_COLORS.get(city, "#636EFA"),
            ))
    apply_common_layout(fig_box, title=f"One-Way ANOVA: {FEATURE_LABELS[anova_feat]} by City")
    fig_box.update_yaxes(title_text=FEATURE_LABELS[anova_feat])
    st.plotly_chart(fig_box, use_container_width=True)

    # ANOVA result
    anova_result = perform_anova(*groups)

    # Compute eta-squared
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum(np.sum((g - grand_mean) ** 2) for g in groups)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("F-statistic", f"{anova_result['f_stat']:.2f}")
    m2.metric("p-value", f"{anova_result['p_value']:.2e}")
    m3.metric("Eta-squared", f"{eta_sq:.4f}")

    if anova_result["p_value"] < 0.05:
        st.success(
            f"ANOVA is significant (F = {anova_result['f_stat']:.2f}, p = {anova_result['p_value']:.2e}). "
            f"At least one city's mean {FEATURE_LABELS[anova_feat]} differs from the others."
        )
    else:
        st.info("ANOVA is not significant. No evidence of differences among city means.")

    # -------------------------------------------------------------------
    # Post-hoc Pairwise Comparisons (Tukey-style)
    # -------------------------------------------------------------------
    st.markdown("#### Okay, Something Differs. But What?")
    st.markdown(
        "ANOVA just told us 'at least one city is different.' Helpful, but not exactly actionable. "
        "It's like a smoke detector going off without telling you which room the fire is in. "
        "That's where post-hoc tests come in. Below we run all pairwise t-tests with Bonferroni "
        "correction to figure out exactly which city pairs are responsible for the overall signal."
    )

    n_comparisons = len(list(combinations(cities_present, 2)))
    alpha_bonf = 0.05 / n_comparisons if n_comparisons > 0 else 0.05

    posthoc_rows = []
    for c1, c2 in combinations(cities_present, 2):
        d1 = fdf[fdf["city"] == c1][anova_feat].dropna()
        d2 = fdf[fdf["city"] == c2][anova_feat].dropna()
        if len(d1) > 0 and len(d2) > 0:
            res = perform_ttest(d1, d2)
            posthoc_rows.append({
                "City 1": c1,
                "City 2": c2,
                "Mean Diff": round(d1.mean() - d2.mean(), 2),
                "t-stat": round(res["t_stat"], 3),
                "p-value": res["p_value"],
                "Cohen's d": round(res["cohens_d"], 3),
                f"Sig (Bonf, alpha={alpha_bonf:.4f})": res["p_value"] < alpha_bonf,
            })

    if posthoc_rows:
        ph_df = pd.DataFrame(posthoc_rows)
        ph_df["p-value"] = ph_df["p-value"].apply(lambda p: f"{p:.2e}")
        st.dataframe(ph_df, use_container_width=True, hide_index=True)
        st.caption(f"Bonferroni-corrected alpha = 0.05 / {n_comparisons} = {alpha_bonf:.4f}")

    # Pairwise mean-difference heatmap
    st.markdown("#### The Difference Map")
    mean_diff_matrix = pd.DataFrame(index=cities_present, columns=cities_present, dtype=float)
    for c1 in cities_present:
        for c2 in cities_present:
            m1_val = fdf[fdf["city"] == c1][anova_feat].mean()
            m2_val = fdf[fdf["city"] == c2][anova_feat].mean()
            mean_diff_matrix.loc[c1, c2] = round(m1_val - m2_val, 2)

    fig_hm = heatmap_chart(
        mean_diff_matrix,
        x_label="City", y_label="City",
        title=f"Pairwise Mean Differences: {FEATURE_LABELS[anova_feat]}",
        color_scale="RdBu_r",
    )
    fig_hm.update_traces(text=mean_diff_matrix.values, texttemplate="%{text:.2f}")
    st.plotly_chart(fig_hm, use_container_width=True)

else:
    st.warning("Need at least 2 cities with data. Adjust sidebar filters.")

st.divider()

# ---------------------------------------------------------------------------
# 3. Two-Way ANOVA: City x Season
# ---------------------------------------------------------------------------
st.subheader("Two-Way ANOVA: When the Answer Is 'It Depends'")

concept_box(
    "Two-Way ANOVA and the Art of Interactions",
    "Two-way ANOVA is where things get genuinely interesting. It tests three questions at once: "
    "(1) Does city matter? (main effect of city) (2) Does season matter? (main effect of season) "
    "and (3) Does the <b>effect of city depend on the season?</b> (the interaction). That third "
    "question is the juicy one. An interaction means the lines on the plot aren't parallel -- "
    "the gap between cities changes across seasons. If NYC and LA have similar temperatures in "
    "summer but wildly different ones in winter, that's an interaction.",
)

tw_feat = st.selectbox(
    "Feature for two-way analysis",
    FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    key="tw_feat",
)

# Compute group means for interaction plot
tw_data = fdf[["city", "season", tw_feat]].dropna()
if len(tw_data) > 0:
    group_means = tw_data.groupby(["season", "city"])[tw_feat].mean().reset_index()

    # Interaction plot
    fig_int = go.Figure()
    for city in cities_present:
        city_means = group_means[group_means["city"] == city]
        # Ensure season order
        city_means = city_means.set_index("season").reindex(SEASON_ORDER).reset_index()
        fig_int.add_trace(go.Scatter(
            x=city_means["season"], y=city_means[tw_feat],
            mode="lines+markers", name=city,
            line=dict(color=CITY_COLORS.get(city, "#636EFA"), width=2),
            marker=dict(size=8),
        ))
    apply_common_layout(fig_int, title=f"Interaction Plot: {FEATURE_LABELS[tw_feat]} by City x Season")
    fig_int.update_xaxes(title_text="Season", categoryorder="array", categoryarray=SEASON_ORDER)
    fig_int.update_yaxes(title_text=f"Mean {FEATURE_LABELS[tw_feat]}")
    st.plotly_chart(fig_int, use_container_width=True)

    # Two-way ANOVA using statsmodels (Type II)
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        tw_data_clean = tw_data.copy()
        tw_data_clean.columns = ["city", "season", "value"]
        model = ols("value ~ C(city) + C(season) + C(city):C(season)", data=tw_data_clean).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        st.markdown("#### The Two-Way ANOVA Table")
        display_table = anova_table.copy()
        display_table.index = display_table.index.str.replace("C(city)", "City", regex=False)
        display_table.index = display_table.index.str.replace("C(season)", "Season", regex=False)
        display_table.index = display_table.index.str.replace(":", " x ", regex=False)
        display_table = display_table.rename(columns={"sum_sq": "Sum of Squares", "df": "df",
                                                       "F": "F-statistic", "PR(>F)": "p-value"})
        st.dataframe(display_table.round(4), use_container_width=True)

        interaction_p = anova_table.loc["C(city):C(season)", "PR(>F)"]
        if interaction_p < 0.05:
            insight_box(
                f"The interaction term is significant (p = {interaction_p:.2e}), which means "
                f"the effect of city on {FEATURE_LABELS[tw_feat]} genuinely depends on the season. "
                "Look at the interaction plot -- the lines aren't parallel. The cities don't maintain "
                "the same relative ordering across all seasons."
            )
        else:
            insight_box(
                f"The interaction is NOT significant (p = {interaction_p:.2e}). The lines in the "
                "interaction plot are roughly parallel, which means the differences between cities "
                "are fairly consistent regardless of season. City and season have independent effects."
            )
    except ImportError:
        st.info("Install statsmodels for the two-way ANOVA table: `pip install statsmodels`")

    # Season-specific ANOVA
    st.markdown("#### Drilling Down: One-Way ANOVA Within Each Season")
    season_anova_rows = []
    for season in SEASON_ORDER:
        s_data = tw_data[tw_data["season"] == season]
        s_groups = [s_data[s_data["city"] == c][tw_feat].dropna().values
                    for c in cities_present if len(s_data[s_data["city"] == c]) > 0]
        s_groups = [g for g in s_groups if len(g) > 0]
        if len(s_groups) >= 2:
            s_res = perform_anova(*s_groups)
            season_anova_rows.append({
                "Season": season,
                "F-statistic": round(s_res["f_stat"], 2),
                "p-value": f"{s_res['p_value']:.2e}",
                "Significant": s_res["p_value"] < 0.05,
            })

    if season_anova_rows:
        st.dataframe(pd.DataFrame(season_anova_rows), use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# 4. Code Example
# ---------------------------------------------------------------------------
code_example("""
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# One-Way ANOVA
groups = [df[df['city'] == c]['temperature_c'].dropna() for c in df['city'].unique()]
f_stat, p_value = stats.f_oneway(*groups)
print(f"F = {f_stat:.2f}, p = {p_value:.2e}")

# Two-Way ANOVA with statsmodels
model = ols('temperature_c ~ C(city) + C(season) + C(city):C(season)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# Post-hoc: Tukey HSD
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(df['temperature_c'], df['city'], alpha=0.05)
print(tukey)
""")

st.divider()

# ---------------------------------------------------------------------------
# 5. Quiz
# ---------------------------------------------------------------------------
quiz(
    "ANOVA's null hypothesis is:",
    [
        "All group means are different",
        "All group means are equal",
        "At least two group means differ",
        "The variances are equal across groups",
    ],
    correct_idx=1,
    explanation="H0 says all the group means are equal: mu_1 = mu_2 = ... = mu_k. The alternative is that at least one differs. Note the asymmetry: ANOVA can tell you 'something is different' but not 'everything is different.' That's what post-hoc tests are for.",
    key="ch14_quiz1",
)

quiz(
    "A significant interaction in two-way ANOVA means:",
    [
        "Both main effects are significant",
        "The factors are independent",
        "The effect of one factor depends on the level of the other",
        "The residuals are not normally distributed",
    ],
    correct_idx=2,
    explanation="An interaction means 'it depends.' The effect of city on temperature changes depending on what season you look at. On the interaction plot, this shows up as non-parallel lines. When people say 'it's complicated,' they usually mean there's an interaction.",
    key="ch14_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 6. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "ANOVA lets you compare 3+ group means with a single F-test, avoiding the multiple-testing disaster of running all pairwise t-tests separately.",
    "The F-statistic is a ratio: between-group variance over within-group variance. If it's large, the group differences can't be explained by random noise alone.",
    "A significant F-test just means 'at least one group differs' -- you still need post-hoc tests (Tukey HSD or Bonferroni-corrected pairwise comparisons) to find out which ones.",
    "Two-way ANOVA adds a second factor and tests for interactions -- the 'it depends' effects where the influence of one factor changes across levels of the other.",
    "Non-parallel lines in an interaction plot are the visual signature of a significant interaction. Parallel lines mean the factors act independently.",
])

navigation(
    prev_label="Ch 13: A/B Testing",
    prev_page="13_AB_Testing.py",
    next_label="Ch 15: Nonparametric Tests",
    next_page="15_Nonparametric_Tests.py",
)
