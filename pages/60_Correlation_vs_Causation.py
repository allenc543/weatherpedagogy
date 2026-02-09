"""Chapter 60: Correlation vs Causation — Confounders, Simpson's paradox, DAGs."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, scatter_chart
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)
from utils.constants import CITY_LIST, CITY_COLORS, FEATURE_LABELS, SEASON_ORDER

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Ch 60: Correlation vs Causation", layout="wide")
df = load_data()
fdf = sidebar_filters(df)

chapter_header(60, "Correlation vs Causation", part="XV")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "Correlation Does Not Imply Causation",
    "Two variables can be correlated (move together) without one <b>causing</b> the other. "
    "Three common reasons for non-causal correlation:<br><br>"
    "<b>1. Confounding</b>: A third variable drives both (e.g., season affects both "
    "temperature and humidity).<br>"
    "<b>2. Reverse causation</b>: The direction of cause is opposite to what we assume.<br>"
    "<b>3. Coincidence</b>: Spurious correlation with no meaningful connection.<br><br>"
    "Establishing causation requires experiments, natural experiments, or careful "
    "causal reasoning with domain knowledge.",
)

st.markdown("""
### Directed Acyclic Graphs (DAGs)

DAGs are diagrams that encode causal assumptions. Arrows represent causal relationships.

```
Season --> Temperature
Season --> Humidity
Temperature <-- Season --> Humidity
```

In this DAG, **Season** is a **confounder** -- it causally affects both temperature
and humidity, creating a correlation between them even though neither directly causes the other.
""")

st.divider()

# ---------------------------------------------------------------------------
# 2. The Temperature-Humidity Correlation
# ---------------------------------------------------------------------------
st.subheader("Case Study: Temperature and Humidity Correlation")

st.markdown(
    "Let us examine the correlation between temperature and humidity in our weather data."
)

# Overall correlation
temp_vals = fdf["temperature_c"].dropna()
hum_vals = fdf.loc[temp_vals.index, "relative_humidity_pct"].dropna()
common_idx = temp_vals.index.intersection(hum_vals.index)
temp_common = fdf.loc[common_idx, "temperature_c"].values
hum_common = fdf.loc[common_idx, "relative_humidity_pct"].values

overall_r, overall_p = stats.pearsonr(temp_common, hum_common)

col_scatter, col_info = st.columns([2, 1])

with col_scatter:
    # Sample for plotting performance
    plot_data = fdf[["temperature_c", "relative_humidity_pct", "city", "season"]].dropna()
    if len(plot_data) > 10000:
        plot_data = plot_data.sample(n=10000, random_state=42)

    fig = scatter_chart(
        plot_data, x="temperature_c", y="relative_humidity_pct",
        color="city", title="Temperature vs Humidity (All Cities)",
        opacity=0.2,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_info:
    st.metric("Overall Correlation (r)", f"{overall_r:.3f}")
    st.metric("p-value", f"{overall_p:.2e}")

    if abs(overall_r) > 0.3:
        st.markdown(
            f"There is a {'negative' if overall_r < 0 else 'positive'} correlation "
            f"of **r = {overall_r:.3f}** between temperature and humidity."
        )
    else:
        st.markdown("The overall correlation is weak.")

    st.markdown("---")
    st.markdown("**Does humidity *cause* temperature?**")
    st.markdown("No! The physical reality is more complex:")
    st.markdown(
        "- Solar radiation heats the surface\n"
        "- Warmer air can hold more moisture (Clausius-Clapeyron)\n"
        "- But relative humidity *decreases* when temp rises faster than moisture\n"
        "- **Season** is the common driver of both"
    )

warning_box(
    "Observing r = {:.3f} between temperature and humidity does NOT mean changing one "
    "will change the other. The correlation exists because both are driven by seasonal "
    "and geographic factors.".format(overall_r)
)

st.divider()

# ---------------------------------------------------------------------------
# 3. Confounding: Season as a Third Variable
# ---------------------------------------------------------------------------
st.subheader("Confounding: The Role of Season")

st.markdown(
    "If season is a confounder, then the correlation between temperature and humidity "
    "should change (often weaken) when we **control for** (condition on) season."
)

# Per-season correlations
season_corrs = []
for season in SEASON_ORDER:
    s_data = fdf[fdf["season"] == season][["temperature_c", "relative_humidity_pct"]].dropna()
    if len(s_data) < 30:
        continue
    r, p = stats.pearsonr(s_data["temperature_c"], s_data["relative_humidity_pct"])
    season_corrs.append({"Season": season, "r": r, "p-value": p, "n": len(s_data)})

season_corr_df = pd.DataFrame(season_corrs)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Correlations by Season")
    display_df = season_corr_df.copy()
    display_df["r"] = display_df["r"].map(lambda x: f"{x:.3f}")
    display_df["p-value"] = display_df["p-value"].map(lambda x: f"{x:.2e}")
    display_df["n"] = display_df["n"].map(lambda x: f"{x:,}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown(f"**Overall correlation: r = {overall_r:.3f}**")

with col2:
    fig_bars = go.Figure()
    fig_bars.add_trace(go.Bar(
        x=season_corr_df["Season"],
        y=season_corr_df["r"],
        marker_color=["#2A9D8F", "#F4A261", "#E63946", "#264653"],
        text=[f"{r:.3f}" for r in season_corr_df["r"]],
        textposition="outside",
    ))
    fig_bars.add_hline(y=overall_r, line_dash="dash", line_color="black",
                       annotation_text=f"Overall: {overall_r:.3f}")
    fig_bars.update_layout(
        xaxis_title="Season", yaxis_title="Pearson r",
    )
    apply_common_layout(fig_bars, title="Correlation by Season vs Overall", height=400)
    st.plotly_chart(fig_bars, use_container_width=True)

# Seasonal scatter plots
fig_seasons = make_subplots(
    rows=1, cols=4,
    subplot_titles=SEASON_ORDER,
)

season_colors = {"Winter": "#2A9D8F", "Spring": "#F4A261", "Summer": "#E63946", "Fall": "#264653"}

for i, season in enumerate(SEASON_ORDER):
    s_data = fdf[fdf["season"] == season][["temperature_c", "relative_humidity_pct"]].dropna()
    if len(s_data) > 3000:
        s_data = s_data.sample(n=3000, random_state=42)

    fig_seasons.add_trace(
        go.Scatter(
            x=s_data["temperature_c"],
            y=s_data["relative_humidity_pct"],
            mode="markers",
            marker=dict(color=season_colors.get(season, "#888"), size=2, opacity=0.3),
            showlegend=False,
        ),
        row=1, col=i + 1,
    )
    fig_seasons.update_xaxes(title_text="Temp (C)", row=1, col=i + 1)
    if i == 0:
        fig_seasons.update_yaxes(title_text="Humidity (%)", row=1, col=i + 1)

fig_seasons.update_layout(template="plotly_white", height=350, margin=dict(t=40, b=40))
st.plotly_chart(fig_seasons, use_container_width=True)

insight_box(
    "When we control for season, the within-season correlations often differ from "
    "the overall correlation. This confirms that season confounds the temperature-humidity "
    "relationship. The overall correlation is partly an artifact of seasonal variation."
)

st.divider()

# ---------------------------------------------------------------------------
# 4. Simpson's Paradox
# ---------------------------------------------------------------------------
st.subheader("Simpson's Paradox with Weather Data")

concept_box(
    "Simpson's Paradox",
    "A trend that appears in aggregated data <b>reverses</b> or disappears when the data "
    "is split into groups. This happens when a lurking variable (confounder) changes the "
    "composition of groups.<br><br>"
    "Example: overall, temperature and humidity might be negatively correlated. "
    "But within each city, the correlation could be positive (or vice versa).",
)

# Per-city correlations
city_corrs = []
for city in CITY_LIST:
    c_data = fdf[fdf["city"] == city][["temperature_c", "relative_humidity_pct"]].dropna()
    if len(c_data) < 30:
        continue
    r, p = stats.pearsonr(c_data["temperature_c"], c_data["relative_humidity_pct"])
    city_corrs.append({"City": city, "r": r, "p-value": p, "n": len(c_data)})

city_corr_df = pd.DataFrame(city_corrs)

col_s1, col_s2 = st.columns(2)

with col_s1:
    st.markdown("#### Per-City Correlations")
    disp = city_corr_df.copy()
    disp["r"] = disp["r"].map(lambda x: f"{x:.3f}")
    disp["p-value"] = disp["p-value"].map(lambda x: f"{x:.2e}")
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.markdown(f"**Overall correlation: r = {overall_r:.3f}**")

with col_s2:
    fig_city_corr = go.Figure()
    fig_city_corr.add_trace(go.Bar(
        x=city_corr_df["City"],
        y=city_corr_df["r"],
        marker_color=[CITY_COLORS.get(c, "#888") for c in city_corr_df["City"]],
        text=[f"{r:.3f}" for r in city_corr_df["r"]],
        textposition="outside",
    ))
    fig_city_corr.add_hline(y=overall_r, line_dash="dash", line_color="black",
                            annotation_text=f"Overall: {overall_r:.3f}")
    fig_city_corr.update_layout(xaxis_title="City", yaxis_title="Pearson r")
    apply_common_layout(fig_city_corr, title="Per-City vs Overall Correlation", height=400)
    st.plotly_chart(fig_city_corr, use_container_width=True)

# Check for sign reversal (Simpson's paradox)
overall_sign = "positive" if overall_r > 0 else "negative"
reversed_cities = [
    row["City"] for _, row in city_corr_df.iterrows()
    if (row["r"] > 0) != (overall_r > 0)
]

if reversed_cities:
    st.error(
        f"**Simpson's Paradox detected!** The overall correlation is {overall_sign} "
        f"(r={overall_r:.3f}), but the following cities show the opposite sign: "
        f"{', '.join(reversed_cities)}."
    )
else:
    st.info(
        "No sign reversal detected between overall and per-city correlations in this case. "
        "Simpson's paradox does not always occur, but it is important to check!"
    )

# Demonstrate with a constructed example
st.markdown("#### Constructed Example: Temperature vs Wind Speed")

st.markdown(
    "Let us check temperature vs wind speed -- a relationship where Simpson's paradox "
    "is more likely due to geographic confounding."
)

tw_data = fdf[["temperature_c", "wind_speed_kmh", "city"]].dropna()
if len(tw_data) > 0:
    tw_overall_r, _ = stats.pearsonr(tw_data["temperature_c"], tw_data["wind_speed_kmh"])
    st.markdown(f"Overall temp-wind correlation: **r = {tw_overall_r:.3f}**")

    tw_city_corrs = []
    for city in CITY_LIST:
        c = tw_data[tw_data["city"] == city]
        if len(c) < 30:
            continue
        r, _ = stats.pearsonr(c["temperature_c"], c["wind_speed_kmh"])
        tw_city_corrs.append({"City": city, "r": f"{r:.3f}"})

    if tw_city_corrs:
        st.dataframe(pd.DataFrame(tw_city_corrs), use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# 5. DAG Visualization
# ---------------------------------------------------------------------------
st.subheader("Causal DAGs for Weather Variables")

st.markdown("""
Below are causal structures we can reason about with weather domain knowledge.
Arrows indicate causal direction.

#### DAG 1: Season Confounds Temperature and Humidity

```
              Season
             /      \\
            v        v
    Temperature    Humidity
```

Season causes both temperature and humidity to change, creating a spurious correlation.

#### DAG 2: Pressure Drives Wind

```
    Pressure Gradient  -->  Wind Speed
```

Differences in atmospheric pressure directly cause wind (physically correct).

#### DAG 3: Temperature - Humidity Chain

```
    Solar Radiation --> Temperature --> Saturation Capacity --> Relative Humidity
```

Temperature does causally affect humidity *capacity*, but relative humidity also
depends on actual moisture content, which has other causes.

#### DAG 4: Full Weather DAG (Simplified)

```
    Solar Radiation  -->  Temperature
         |                    |
         v                    v
    Evaporation  -->  Specific Humidity  -->  Relative Humidity
                                              ^
                                              |
                                          Temperature
                                          (denominator)
```
""")

insight_box(
    "DAGs encode our assumptions about causal structure. They help us identify "
    "confounders (backdoor paths), mediators, and colliders. In weather science, "
    "physical mechanisms provide strong prior knowledge about causal direction."
)

st.divider()

# ---------------------------------------------------------------------------
# 6. Interactive: Control for Confounders
# ---------------------------------------------------------------------------
st.subheader("Interactive: Control for Confounders")

st.markdown(
    "Select two variables and a potential confounder. We show the correlation "
    "before and after controlling for the confounder."
)

conf_cols = ["temperature_c", "relative_humidity_pct", "wind_speed_kmh", "surface_pressure_hpa"]
col_c1, col_c2 = st.columns(2)
with col_c1:
    var1 = st.selectbox("Variable 1", conf_cols,
                        format_func=lambda c: FEATURE_LABELS.get(c, c),
                        index=0, key="cc_var1")
    var2 = st.selectbox("Variable 2", conf_cols,
                        format_func=lambda c: FEATURE_LABELS.get(c, c),
                        index=1, key="cc_var2")
with col_c2:
    confounder_type = st.selectbox(
        "Confounder to control for",
        ["Season", "Month", "City", "Hour of Day"],
        key="cc_conf",
    )

conf_data = fdf[[var1, var2, "season", "month", "city", "hour", "month_name"]].dropna()

# Overall correlation
r_overall, _ = stats.pearsonr(conf_data[var1], conf_data[var2])

# Group variable
group_map = {
    "Season": "season",
    "Month": "month_name",
    "City": "city",
    "Hour of Day": "hour",
}
group_col = group_map[confounder_type]

# Per-group correlations
group_corrs = []
for group in sorted(conf_data[group_col].unique()):
    g_data = conf_data[conf_data[group_col] == group]
    if len(g_data) < 30:
        continue
    r, p = stats.pearsonr(g_data[var1], g_data[var2])
    group_corrs.append({"Group": str(group), "r": r, "n": len(g_data)})

group_corr_df = pd.DataFrame(group_corrs)

# Partial correlation (using residuals)
from numpy.linalg import lstsq

# Encode confounder as dummy variables
conf_dummies = pd.get_dummies(conf_data[group_col], drop_first=True).values
if conf_dummies.shape[1] > 0:
    A = np.column_stack([np.ones(len(conf_data)), conf_dummies])
    # Residualize var1
    b1, _, _, _ = lstsq(A, conf_data[var1].values, rcond=None)
    res1 = conf_data[var1].values - A @ b1
    # Residualize var2
    b2, _, _, _ = lstsq(A, conf_data[var2].values, rcond=None)
    res2 = conf_data[var2].values - A @ b2
    r_partial, _ = stats.pearsonr(res1, res2)
else:
    r_partial = r_overall

col_res1, col_res2 = st.columns(2)

with col_res1:
    st.metric("Overall Correlation", f"r = {r_overall:.3f}")
    st.metric(f"Partial Correlation (controlling {confounder_type})", f"r = {r_partial:.3f}")
    st.metric("Change in |r|", f"{abs(r_partial) - abs(r_overall):.3f}")

    if abs(r_partial) < abs(r_overall) * 0.8:
        st.success(
            f"Controlling for {confounder_type} **reduces** the correlation, "
            "suggesting it is a confounder!"
        )
    else:
        st.info(
            f"Controlling for {confounder_type} does not substantially reduce "
            "the correlation. It may not be a strong confounder for this pair."
        )

with col_res2:
    if len(group_corr_df) > 0:
        fig_gc = go.Figure()
        fig_gc.add_trace(go.Bar(
            x=group_corr_df["Group"].astype(str),
            y=group_corr_df["r"],
            marker_color="#2A9D8F",
            text=[f"{r:.3f}" for r in group_corr_df["r"]],
            textposition="outside",
        ))
        fig_gc.add_hline(y=r_overall, line_dash="dash", line_color="#E63946",
                         annotation_text=f"Overall: {r_overall:.3f}")
        fig_gc.update_layout(
            xaxis_title=confounder_type, yaxis_title="Correlation (r)",
        )
        apply_common_layout(fig_gc, title=f"Correlation by {confounder_type}", height=400)
        st.plotly_chart(fig_gc, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 7. Code Example
# ---------------------------------------------------------------------------
code_example("""
import numpy as np
import pandas as pd
from scipy import stats

# Overall correlation
r_overall, p = stats.pearsonr(df['temperature_c'], df['relative_humidity_pct'])
print(f"Overall: r={r_overall:.3f}, p={p:.2e}")

# Per-season correlation (controlling for season)
for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    s = df[df['season'] == season]
    r, p = stats.pearsonr(s['temperature_c'], s['relative_humidity_pct'])
    print(f"{season}: r={r:.3f}")

# Partial correlation using residuals
from numpy.linalg import lstsq

# Create season dummies
dummies = pd.get_dummies(df['season'], drop_first=True).values
A = np.column_stack([np.ones(len(df)), dummies])

# Residualise both variables
b1, _, _, _ = lstsq(A, df['temperature_c'].values, rcond=None)
res_temp = df['temperature_c'].values - A @ b1

b2, _, _, _ = lstsq(A, df['relative_humidity_pct'].values, rcond=None)
res_hum = df['relative_humidity_pct'].values - A @ b2

r_partial, _ = stats.pearsonr(res_temp, res_hum)
print(f"Partial correlation (controlling season): r={r_partial:.3f}")
""")

st.divider()

# ---------------------------------------------------------------------------
# 8. Quiz
# ---------------------------------------------------------------------------
quiz(
    "If temperature and humidity are correlated with r = -0.5, can we conclude "
    "that increasing temperature causes humidity to decrease?",
    [
        "Yes, the strong correlation proves causation",
        "Yes, because the p-value would be significant",
        "No, the correlation could be driven by a confounder like season",
        "No, because the correlation is negative",
    ],
    correct_idx=2,
    explanation="Correlation does not imply causation. A confounder like season can "
                "create a correlation between variables without a direct causal link.",
    key="ch60_quiz1",
)

quiz(
    "In Simpson's paradox:",
    [
        "The sample size is too small to detect a real effect",
        "A trend in aggregated data reverses when the data is split by a grouping variable",
        "Two variables are perfectly correlated",
        "The data contains missing values",
    ],
    correct_idx=1,
    explanation="Simpson's paradox occurs when a trend in aggregated data reverses or "
                "disappears in subgroups, usually due to a confounding variable.",
    key="ch60_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 9. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Correlation measures association, not causation. Confounders can create spurious correlations.",
    "DAGs encode causal assumptions and help identify confounders, mediators, and colliders.",
    "Controlling for a confounder (via stratification or partial correlation) can reveal the true relationship.",
    "Simpson's paradox shows that aggregate trends can reverse in subgroups.",
    "Weather provides excellent examples: season confounds many variable relationships.",
])

navigation(
    prev_label="Ch 59: Autoencoder Anomaly Detection",
    prev_page="59_Autoencoder_Anomaly_Detection.py",
    next_label="Ch 61: Natural Experiments",
    next_page="61_Natural_Experiments.py",
)
