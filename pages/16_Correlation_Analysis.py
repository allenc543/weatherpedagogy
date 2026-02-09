"""Chapter 16: Correlation Analysis — Pearson, Spearman, partial correlation, spurious correlation."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as sp_stats

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import (
    apply_common_layout, color_map, scatter_chart, heatmap_chart,
)
from utils.stats_helpers import correlation_matrix
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)
from utils.constants import CITY_LIST, FEATURE_COLS, FEATURE_LABELS, CITY_COLORS

# ---------------------------------------------------------------------------
df = load_data()
fdf = sidebar_filters(df)

chapter_header(16, "Correlation Analysis", part="IV")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "Correlation: The Most Misused Statistic in the World",
    "Correlation measures how two variables move together, and it ranges from -1 (perfect inverse "
    "relationship) through 0 (no relationship) to +1 (perfect lockstep). It's one of the most "
    "intuitive statistics out there, which is precisely what makes it dangerous. People see a "
    "correlation of 0.7 and immediately start thinking about causation. They shouldn't. Ice cream "
    "sales and drowning deaths are correlated (both go up in summer), but buying ice cream doesn't "
    "make you drown. <b>Correlation does not imply causation</b> -- and we'll see exactly why "
    "later in this chapter when we get to confounders.",
)

col1, col2 = st.columns(2)
with col1:
    formula_box(
        "Pearson Correlation (r)",
        r"r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}"
        r"{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}",
        "Measures LINEAR association. If the relationship is curved (like, say, temperature vs day-of-year), Pearson will underestimate it. Also sensitive to outliers -- one weird data point can move r substantially.",
    )
with col2:
    formula_box(
        "Spearman Rank Correlation (rho)",
        r"\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}",
        "Measures MONOTONIC association using ranks. If one variable always increases when the other does (even non-linearly), Spearman will catch it. More robust to outliers because it only cares about ordering, not magnitude.",
    )

st.markdown("### How Strong Is That Correlation, Really?")
strength_df = pd.DataFrame({
    "|r|": ["0.00 - 0.19", "0.20 - 0.39", "0.40 - 0.59", "0.60 - 0.79", "0.80 - 1.00"],
    "Interpretation": ["Negligible", "Weak", "Moderate", "Strong", "Very Strong"],
})
st.dataframe(strength_df, use_container_width=True, hide_index=True)

warning_box(
    "Pearson catches only linear relationships; Spearman catches only monotonic ones. "
    "Two variables can have a perfectly clear relationship -- a U-shape, a sine wave, whatever "
    "-- and still produce r close to 0. A low correlation means 'no linear/monotonic relationship,' "
    "NOT 'no relationship at all.' Always look at the scatter plot."
)

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Correlation Between Two Features
# ---------------------------------------------------------------------------
st.subheader("Explore: Pick Two Variables and See What Happens")

ctrl_col, plot_col = st.columns([1, 2])

with ctrl_col:
    feat_x = st.selectbox(
        "X-axis feature",
        FEATURE_COLS,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
        index=0,
        key="corr_x",
    )
    feat_y = st.selectbox(
        "Y-axis feature",
        FEATURE_COLS,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
        index=1,
        key="corr_y",
    )
    corr_method = st.radio("Correlation method", ["Pearson", "Spearman"], key="corr_method")
    color_by = st.radio("Color by", ["City", "Season"], key="corr_color")

color_col = "city" if color_by == "City" else "season"
plot_data = fdf[[feat_x, feat_y, "city", "season"]].dropna()

if len(plot_data) > 10000:
    plot_sample = plot_data.sample(10000, random_state=42)
else:
    plot_sample = plot_data

with plot_col:
    if color_by == "City":
        fig = scatter_chart(
            plot_sample, x=feat_x, y=feat_y, color="city",
            title=f"{FEATURE_LABELS[feat_x]} vs {FEATURE_LABELS[feat_y]}",
            opacity=0.3,
        )
    else:
        import plotly.express as px
        fig = px.scatter(
            plot_sample, x=feat_x, y=feat_y, color="season",
            opacity=0.3,
            labels={feat_x: FEATURE_LABELS[feat_x], feat_y: FEATURE_LABELS[feat_y]},
            title=f"{FEATURE_LABELS[feat_x]} vs {FEATURE_LABELS[feat_y]}",
        )
        apply_common_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

# Overall correlation
method_lower = corr_method.lower()
r_overall, p_overall = sp_stats.pearsonr(plot_data[feat_x], plot_data[feat_y]) if method_lower == "pearson" \
    else sp_stats.spearmanr(plot_data[feat_x], plot_data[feat_y])

st.markdown(f"**Overall {corr_method} correlation:** r = {r_overall:.4f} (p = {p_overall:.2e})")

st.divider()

# ---------------------------------------------------------------------------
# 3. Correlation by City
# ---------------------------------------------------------------------------
st.subheader("The Same Correlation Can Tell Very Different Stories")

st.markdown(
    "Here's something that catches people off guard: the overall correlation can be completely "
    "different from the within-group correlations. This is Simpson's Paradox territory. "
    "Let's compute the correlation separately for each city and see if the story changes."
)

city_corr_rows = []
for city in sorted(fdf["city"].unique()):
    cdata = fdf[fdf["city"] == city][[feat_x, feat_y]].dropna()
    if len(cdata) > 10:
        if method_lower == "pearson":
            r, p = sp_stats.pearsonr(cdata[feat_x], cdata[feat_y])
        else:
            r, p = sp_stats.spearmanr(cdata[feat_x], cdata[feat_y])
        city_corr_rows.append({
            "City": city,
            f"{corr_method} r": round(r, 4),
            "p-value": f"{p:.2e}",
            "n": len(cdata),
        })

if city_corr_rows:
    city_corr_df = pd.DataFrame(city_corr_rows)
    st.dataframe(city_corr_df, use_container_width=True, hide_index=True)

    # Bar chart of correlations by city
    fig_bar = go.Figure()
    for _, row in city_corr_df.iterrows():
        fig_bar.add_trace(go.Bar(
            x=[row["City"]], y=[row[f"{corr_method} r"]],
            marker_color=CITY_COLORS.get(row["City"], "#636EFA"),
            name=row["City"],
            showlegend=False,
        ))
    fig_bar.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
    apply_common_layout(
        fig_bar,
        title=f"{corr_method} Correlation ({FEATURE_LABELS[feat_x]} vs {FEATURE_LABELS[feat_y]}) by City",
    )
    fig_bar.update_yaxes(title_text=f"{corr_method} r", range=[-1, 1])
    st.plotly_chart(fig_bar, use_container_width=True)

    insight_box(
        "Notice how the correlation can vary substantially across cities. Coastal cities like "
        "Houston, where the ocean moderates both temperature and humidity, may show a very "
        "different temperature-humidity relationship than inland cities like Dallas, where "
        "continental climate dynamics dominate. The overall correlation is an average of these "
        "different stories -- and sometimes an average tells you nothing useful about any "
        "individual case."
    )

st.divider()

# ---------------------------------------------------------------------------
# 4. Full Correlation Matrix
# ---------------------------------------------------------------------------
st.subheader("The Full Correlation Matrix: Everything Correlated With Everything")

matrix_method = st.radio("Method for matrix", ["pearson", "spearman"], key="mat_method")

all_cities = sorted(fdf["city"].unique())
selected_city_for_matrix = st.selectbox(
    "City (or 'All' for combined)", ["All"] + all_cities, key="mat_city"
)

if selected_city_for_matrix == "All":
    mat_data = fdf[FEATURE_COLS]
else:
    mat_data = fdf[fdf["city"] == selected_city_for_matrix][FEATURE_COLS]

corr_mat = correlation_matrix(mat_data, method=matrix_method)
corr_mat.columns = [FEATURE_LABELS.get(c, c) for c in corr_mat.columns]
corr_mat.index = [FEATURE_LABELS.get(c, c) for c in corr_mat.index]

fig_hm = heatmap_chart(
    corr_mat,
    title=f"{matrix_method.title()} Correlation Matrix ({selected_city_for_matrix})",
    color_scale="RdBu_r",
)
fig_hm.update_traces(text=corr_mat.values.round(3), texttemplate="%{text}")
fig_hm.update_layout(height=500)
st.plotly_chart(fig_hm, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 5. Seasonal Confounding & Partial Correlation
# ---------------------------------------------------------------------------
st.subheader("The Plot Twist: Spurious Correlation and Confounders")

concept_box(
    "When Correlation Lies: Confounding Variables",
    "Imagine you discover that temperature and humidity are strongly correlated. 'Aha!' you think, "
    "'hot weather causes high humidity!' But wait. Both temperature AND humidity change with the "
    "<b>seasons</b>. Summer brings heat AND Gulf moisture. Maybe the correlation is just reflecting "
    "the fact that both variables dance to the same seasonal tune, not that one causes the other. "
    "This is a <b>confounder</b> -- a third variable driving both of the ones you're looking at. "
    "<b>Partial correlation</b> is the fix: it mathematically removes the confounder's influence "
    "to reveal whatever relationship remains underneath.",
)

formula_box(
    "Partial Correlation",
    r"r_{xy \cdot z} = \frac{r_{xy} - r_{xz} \cdot r_{yz}}"
    r"{\sqrt{(1 - r_{xz}^2)(1 - r_{yz}^2)}}",
    "The correlation between x and y after subtracting out whatever z explains about each of them. If this drops to near zero, the original correlation was largely an artifact of the confounder.",
)

# Demonstrate with temperature vs humidity, controlling for month
st.markdown("#### Case Study: Temperature vs Humidity, Controlling for Month")

pc_data = fdf[["temperature_c", "relative_humidity_pct", "month"]].dropna()

if len(pc_data) > 10:
    # Raw correlation
    r_raw, _ = sp_stats.pearsonr(pc_data["temperature_c"], pc_data["relative_humidity_pct"])

    # Partial correlation controlling for month
    r_xz, _ = sp_stats.pearsonr(pc_data["temperature_c"], pc_data["month"])
    r_yz, _ = sp_stats.pearsonr(pc_data["relative_humidity_pct"], pc_data["month"])
    r_partial = (r_raw - r_xz * r_yz) / np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))

    pcol1, pcol2 = st.columns(2)
    pcol1.metric("Raw Pearson r (Temp vs Humidity)", f"{r_raw:.4f}")
    pcol2.metric("Partial r (controlling for Month)", f"{r_partial:.4f}")

    diff = abs(r_raw) - abs(r_partial)
    if abs(diff) > 0.05:
        insight_box(
            f"After controlling for month, the correlation changes from {r_raw:.4f} to {r_partial:.4f}. "
            "That shift tells us that some of what looked like a direct temperature-humidity relationship "
            "was actually seasonal variation pulling both variables in the same direction. The partial "
            "correlation reveals the relationship that exists above and beyond seasonal patterns."
        )
    else:
        insight_box(
            f"The partial correlation ({r_partial:.4f}) is quite similar to the raw correlation ({r_raw:.4f}). "
            "This means month isn't really confounding this relationship -- temperature and humidity "
            "are associated even within the same month. The relationship is genuine, not just a "
            "seasonal artifact."
        )

    # Show within-season correlations
    st.markdown("#### Breaking It Down by Season")
    from utils.constants import SEASON_ORDER
    season_corrs = []
    for season in SEASON_ORDER:
        s_data = fdf[fdf["season"] == season][["temperature_c", "relative_humidity_pct"]].dropna()
        if len(s_data) > 10:
            r, p = sp_stats.pearsonr(s_data["temperature_c"], s_data["relative_humidity_pct"])
            season_corrs.append({"Season": season, "Pearson r": round(r, 4), "p-value": f"{p:.2e}", "n": len(s_data)})

    if season_corrs:
        st.dataframe(pd.DataFrame(season_corrs), use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# 6. Code Example
# ---------------------------------------------------------------------------
code_example("""
from scipy import stats
import pandas as pd

# Pearson correlation
r, p = stats.pearsonr(df['temperature_c'], df['relative_humidity_pct'])
print(f"Pearson r = {r:.4f}, p = {p:.2e}")

# Spearman correlation
rho, p = stats.spearmanr(df['temperature_c'], df['relative_humidity_pct'])
print(f"Spearman rho = {rho:.4f}, p = {p:.2e}")

# Correlation matrix
corr = df[['temperature_c', 'relative_humidity_pct',
           'wind_speed_kmh', 'surface_pressure_hpa']].corr()
print(corr)

# Partial correlation (controlling for month)
r_xy = stats.pearsonr(df['temperature_c'], df['relative_humidity_pct'])[0]
r_xz = stats.pearsonr(df['temperature_c'], df['month'])[0]
r_yz = stats.pearsonr(df['relative_humidity_pct'], df['month'])[0]
r_partial = (r_xy - r_xz * r_yz) / ((1 - r_xz**2) * (1 - r_yz**2))**0.5
print(f"Partial r (controlling for month) = {r_partial:.4f}")
""")

st.divider()

# ---------------------------------------------------------------------------
# 7. Quiz
# ---------------------------------------------------------------------------
quiz(
    "A Pearson correlation of -0.85 indicates:",
    [
        "A weak negative relationship",
        "A strong negative linear relationship",
        "No relationship",
        "A strong positive relationship",
    ],
    correct_idx=1,
    explanation="The magnitude (0.85) puts this squarely in 'very strong' territory, and the negative sign means the variables move in opposite directions. As one goes up, the other tends to go down -- strongly and consistently.",
    key="ch16_quiz1",
)

quiz(
    "Two variables have r = 0.70, but after controlling for season, the partial correlation drops to 0.10. What does this suggest?",
    [
        "The variables are causally related",
        "The correlation was largely driven by the seasonal confound",
        "Spearman should be used instead",
        "The sample size is too small",
    ],
    correct_idx=1,
    explanation="When the correlation collapses after controlling for a third variable, the original association was mostly an artifact. Both variables were dancing to the season's tune, creating the illusion of a direct relationship. This is exactly the kind of thing partial correlation is designed to unmask.",
    key="ch16_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 8. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Pearson measures linear correlation; Spearman measures monotonic rank correlation. Neither captures non-monotonic relationships. When in doubt, look at the scatter plot.",
    "Correlation strength: |r| < 0.2 is negligible (basically noise), 0.4-0.6 is moderate (something's going on), > 0.8 is very strong (hard to ignore). But these are guidelines, not gospel.",
    "Correlation does NOT imply causation. This gets repeated so often it's become a cliche, but people still fall for it constantly. Ice cream and drowning, anyone?",
    "Partial correlation is your tool for unmasking confounders. If the correlation collapses after controlling for a third variable, the original association was probably spurious.",
    "The same two features can show wildly different correlations in different cities or seasons. Aggregated correlations can hide, or even reverse, the patterns within subgroups.",
])

navigation(
    prev_label="Ch 15: Nonparametric Tests",
    prev_page="15_Nonparametric_Tests.py",
    next_label="Ch 17: Simple Linear Regression",
    next_page="17_Simple_Linear_Regression.py",
)
