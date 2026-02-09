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
df = load_data()
fdf = sidebar_filters(df)

chapter_header(60, "Correlation vs Causation", part="XV")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
st.markdown(
    "Let me start with a specific finding from our weather data, and then explain why "
    "it does not mean what it appears to mean."
)
st.markdown(
    "**The finding**: Across our 105,000 hourly readings from 6 cities, temperature and "
    "relative humidity are correlated. When temperature goes up, humidity tends to go down "
    "(or vice versa, depending on which cities and time ranges you include). You compute "
    "a Pearson correlation, get a number like r = -0.45, and the p-value is essentially zero."
)
st.markdown(
    "**The tempting conclusion**: 'Aha! Raising the temperature causes humidity to drop. "
    "If we could heat up the air, we could reduce humidity.' This sounds plausible -- and "
    "it is exactly the kind of reasoning that gets people into trouble."
)
st.markdown(
    "**The actual explanation**: Both temperature and humidity are driven by **season**. "
    "In summer, the sun heats the air (high temperature) and the relative humidity drops "
    "because warm air can hold more moisture without feeling 'humid.' In winter, temperatures "
    "fall and relative humidity rises. Season is moving both variables simultaneously, "
    "creating a correlation between them -- but temperature is not *causing* humidity to "
    "change any more than wearing shorts causes ice cream sales to increase."
)

concept_box(
    "Three Reasons Variables Can Be Correlated Without One Causing the Other",
    "Our weather data illustrates all three patterns:<br><br>"
    "<b>1. Confounding (the big one)</b>: A third variable drives both. Season affects both "
    "temperature and humidity, creating a correlation between them. If you look at July "
    "data alone (holding season constant), the temperature-humidity correlation might be "
    "completely different. The confounder is doing all the work.<br><br>"
    "<b>2. Reverse causation</b>: You might observe that high humidity co-occurs with high "
    "temperature and conclude humidity causes warming. Actually, the physical causation runs "
    "the other way -- temperature affects the air's capacity for moisture, which affects "
    "relative humidity. Getting the arrow direction wrong leads to wrong interventions.<br><br>"
    "<b>3. Spurious correlation (coincidence)</b>: In a dataset of 105,000 rows across "
    "6 cities, you will find correlations between variables that have no meaningful "
    "connection. If NYC happens to have high wind in months when Dallas has high pressure, "
    "the correlation is real but meaningless. With enough variables, some will correlate "
    "by pure chance.",
)

st.markdown("""
### Directed Acyclic Graphs (DAGs)

The tool that makes causal reasoning rigorous is the **DAG** -- a diagram where arrows
represent causal relationships. Here is what the temperature-humidity situation really
looks like:

```
              Season
             /      \\
            v        v
    Temperature    Humidity
```

**Season** is a **confounder**. It causally affects both temperature and humidity.
The arrow from Season to Temperature says 'season causes temperature changes' (true -- it is
hotter in summer). The arrow from Season to Humidity says 'season causes humidity changes'
(also true). But there is NO arrow directly between Temperature and Humidity -- the
correlation between them is entirely an artifact of their shared cause.

This is not just an academic point. If you built a model to predict humidity from temperature
without accounting for season, your model would work fine on historical data (the correlation
is real!) but would give completely wrong answers if you tried to use it for intervention
('what would happen to humidity if we artificially raised the temperature?').
""")

st.divider()

# ---------------------------------------------------------------------------
# 2. The Temperature-Humidity Correlation
# ---------------------------------------------------------------------------
st.subheader("Case Study: Temperature and Humidity Correlation")

st.markdown(
    "Let us see the correlation in our actual data. This scatter plot shows every "
    "temperature-humidity pair (subsampled for performance), color-coded by city."
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
    st.markdown("**Does humidity *cause* temperature changes?**")
    st.markdown(
        "No. The physical reality is that solar radiation heats the surface "
        "(raising temperature), and warmer air has greater moisture-holding capacity, "
        "which *lowers* relative humidity even if absolute moisture stays the same. "
        "Meanwhile, **season** drives both: summer brings heat and changes the moisture "
        "balance, winter does the opposite. The correlation is real, but the causal story "
        "is not 'change temperature to change humidity.'"
    )

warning_box(
    "Observing r = {:.3f} between temperature and humidity does NOT mean that "
    "intervening on one will change the other. The correlation exists because both "
    "are driven by seasonal and geographic factors. If you conditioned on 'July in "
    "Houston,' the relationship between these variables could be completely different.".format(overall_r)
)

st.divider()

# ---------------------------------------------------------------------------
# 3. Confounding: Season as a Third Variable
# ---------------------------------------------------------------------------
st.subheader("Confounding: The Role of Season")

st.markdown(
    "Here is the test. If season really is confounding the temperature-humidity correlation, "
    "then the correlation should change -- often weaken or even reverse -- when we look "
    "at each season separately. This is what 'controlling for a confounder' means: you hold "
    "the confounder constant and see if the relationship between the other two variables "
    "still holds."
)
st.markdown(
    "Think of it this way: the overall correlation mixes together summer data (hot, low-ish "
    "humidity) and winter data (cold, high-ish humidity). That mixing creates a negative "
    "slope. But *within* summer alone, or *within* winter alone, the relationship might be "
    "much weaker or even positive, because you have removed the between-season variation "
    "that was driving the correlation."
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
    "Compare the within-season correlations to the overall. If the within-season correlations "
    "are weaker (closer to zero) or even opposite in sign compared to the overall r, that "
    "confirms season is confounding the relationship. The overall correlation was being driven "
    "by comparing 'cold winter days with high humidity' against 'hot summer days with low "
    "humidity' -- not by any within-season relationship between temperature and humidity. "
    "This is the single most important lesson in causal inference: the overall pattern can "
    "be an illusion created by a lurking variable."
)

st.divider()

# ---------------------------------------------------------------------------
# 4. Simpson's Paradox
# ---------------------------------------------------------------------------
st.subheader("Simpson's Paradox with Weather Data")

concept_box(
    "Simpson's Paradox: When the Aggregate Lies",
    "Simpson's paradox is the extreme version of confounding. A trend that appears in "
    "the combined data <b>completely reverses</b> when you split by a grouping variable.<br><br>"
    "Here is a concrete weather example. Suppose the overall correlation between temperature "
    "and humidity across all 6 cities is negative (r = -0.45). But when you look at each city "
    "individually, the correlation within every city is <em>positive</em>. How is that possible?<br><br>"
    "Because the cities have different baselines. Los Angeles is cool and dry. Houston is hot "
    "and humid. Dallas is hot and moderate. When you mix them together, you are comparing "
    "'cool dry LA days' against 'hot humid Houston days,' creating a negative slope. But "
    "within each city, warmer days might actually be more humid (convective moisture). "
    "The aggregate trend is the opposite of every subgroup trend. That is Simpson's paradox.",
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
        f"{', '.join(reversed_cities)}. This means the aggregate trend is misleading -- "
        "city-level geography is confounding the relationship."
    )
else:
    st.info(
        "No sign reversal detected between overall and per-city correlations in this case. "
        "Simpson's paradox does not always occur, but the per-city correlations still differ "
        "from the overall, confirming that city is a confounder even without full reversal."
    )

# Demonstrate with a constructed example
st.markdown("#### Exploring Another Pair: Temperature vs Wind Speed")

st.markdown(
    "Let us check temperature vs wind speed -- a relationship where Simpson's paradox "
    "is more likely because different cities have very different wind patterns. Coastal "
    "cities (Houston, LA) get sea breezes that inland cities (Dallas, Austin) do not, "
    "creating geographic confounding."
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
DAGs make our causal assumptions explicit and testable. Here are four DAGs for
relationships in our weather data, each grounded in meteorological physics.

#### DAG 1: Season Confounds Temperature and Humidity

```
              Season
             /      \\
            v        v
    Temperature    Humidity
```

This is our running example. Season drives both variables. The temperature-humidity
correlation is not causal -- it is an artifact of the shared seasonal driver. To test
this, we control for season and check if the correlation weakens.

#### DAG 2: Pressure Gradient Drives Wind

```
    Pressure Gradient  -->  Wind Speed
```

This one IS causal. Differences in atmospheric pressure directly cause air to move (wind).
This is one of the most well-established physical laws in meteorology. We will test this
with Granger causality in the next chapter.

#### DAG 3: Temperature Affects Humidity Capacity

```
    Solar Radiation --> Temperature --> Saturation Capacity --> Relative Humidity
```

Temperature does causally affect humidity *capacity* (the Clausius-Clapeyron equation).
But relative humidity also depends on actual moisture content, which has other causes
(proximity to ocean, recent rainfall, air mass origin). So the causal chain exists but
is more complex than a simple 'temperature causes humidity.'

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

Temperature appears twice -- once as a cause of evaporation and once as the denominator
in the relative humidity calculation. This is why the relationship is complex: temperature
affects relative humidity through two pathways that push in opposite directions.
""")

insight_box(
    "DAGs are not decorations -- they are tools for reasoning about which variables you "
    "need to control for (and which you must NOT control for, because doing so can introduce "
    "bias). In our weather data, controlling for season removes confounding. But if you "
    "accidentally controlled for a variable that sits on the causal pathway (like specific "
    "humidity between temperature and relative humidity), you would block the real causal "
    "signal. DAGs help you avoid these mistakes by making the causal structure explicit."
)

st.divider()

# ---------------------------------------------------------------------------
# 6. Interactive: Control for Confounders
# ---------------------------------------------------------------------------
st.subheader("Interactive: Control for Confounders")

st.markdown(
    "Now you can test any pair of weather variables and see what happens when you control "
    "for a potential confounder. The 'overall correlation' is what you see in the raw data. "
    "The 'partial correlation' is what remains after removing the confounder's influence "
    "(by regressing both variables on the confounder and correlating the residuals). "
    "If the partial correlation is much weaker, the confounder was driving the relationship."
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
            f"Controlling for {confounder_type} **reduces** the correlation from "
            f"|{r_overall:.3f}| to |{r_partial:.3f}|. This suggests {confounder_type} "
            "is confounding the relationship -- it was driving at least part of the "
            "observed correlation."
        )
    else:
        st.info(
            f"Controlling for {confounder_type} does not substantially reduce "
            "the correlation. It may not be a strong confounder for this variable pair. "
            "Try a different confounder -- or the relationship may be genuinely direct."
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
    "In our weather data, temperature and humidity are correlated with r = -0.45 overall. "
    "But within each season, the correlation is only r = -0.15. What does this tell us?",
    [
        "The overall correlation is wrong and should be discarded",
        "Season is confounding the relationship -- much of the r = -0.45 was driven by "
        "comparing hot summers against cold winters, not by a within-season relationship",
        "The seasonal data has too few samples to be reliable",
        "Humidity must be causing temperature changes",
    ],
    correct_idx=1,
    explanation="The drop from r = -0.45 to r = -0.15 when controlling for season is the "
                "smoking gun of confounding. The overall correlation mixed together summer "
                "(hot, drier) and winter (cold, more humid) data, creating an artificially strong "
                "negative slope. Within any single season, temperature and humidity vary together "
                "much less strongly, because you have removed the between-season variation that "
                "was driving the correlation. The r = -0.15 within-season is the 'true' "
                "within-season relationship; the r = -0.45 was inflated by the confounder.",
    key="ch60_quiz1",
)

quiz(
    "You find that the overall correlation between temperature and wind speed is r = 0.10 "
    "across all cities. But in Los Angeles, the correlation is r = -0.25, and in Dallas it "
    "is r = 0.30. This is an example of:",
    [
        "Measurement error in the wind speed sensor",
        "Simpson's paradox -- the subgroup trends differ from (and in one case reverse) "
        "the aggregate trend due to geographic confounding",
        "A normal statistical fluctuation with no deeper meaning",
        "Evidence that temperature causes wind",
    ],
    correct_idx=1,
    explanation="This is Simpson's paradox in action. Los Angeles gets sea breezes that bring "
                "cool air and wind from the ocean -- cooler days are windier there (negative "
                "correlation). Dallas, far inland, gets hot dry winds from the southwest in summer "
                "-- hotter days are windier there (positive correlation). When you mix the two "
                "cities, the opposing effects partially cancel, giving a weak overall r = 0.10 "
                "that hides two strong but opposite city-level patterns. The aggregate number is "
                "technically correct but misleading -- the real story is at the city level.",
    key="ch60_quiz2",
)

quiz(
    "A data scientist builds a model predicting humidity from temperature alone. It achieves "
    "R-squared = 0.20 on test data. They conclude: 'temperature explains 20% of humidity "
    "variation.' Is this interpretation correct?",
    [
        "Yes -- R-squared directly measures causal explanation",
        "No -- R-squared measures predictive association, not causal explanation. Much of that "
        "20% might come from season confounding both variables, not from temperature directly "
        "affecting humidity",
        "No -- 20% is too low to draw any conclusions",
        "Yes -- because the model used test data, not training data",
    ],
    correct_idx=1,
    explanation="R-squared measures how well temperature *predicts* humidity, which includes "
                "both direct causal effects and indirect paths through confounders. If season "
                "drives both variables, then temperature can predict humidity (because both "
                "co-vary with season) without temperature actually *causing* humidity changes. "
                "To estimate the causal effect of temperature on humidity, you would need to "
                "control for season (and other confounders) first. The partial R-squared after "
                "controlling for confounders would be the right measure of the direct relationship.",
    key="ch60_quiz3",
)

st.divider()

# ---------------------------------------------------------------------------
# 9. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Correlation measures statistical association, not causation. In our weather data, "
    "temperature and humidity correlate at r = {:.3f} -- but season drives both. Controlling "
    "for season reveals a much weaker (or different) relationship.".format(overall_r),
    "Confounders are third variables that drive both the 'cause' and the 'effect.' Season is "
    "the classic weather confounder: it makes temperature and humidity move together without "
    "either one directly causing changes in the other.",
    "Simpson's paradox happens when aggregate trends reverse in subgroups. Pooling data across "
    "cities with different climates can create overall correlations that no individual city "
    "exhibits. Always check subgroups before trusting an aggregate trend.",
    "DAGs make causal assumptions explicit. Drawing arrows between weather variables forces "
    "you to state your causal beliefs and identifies which confounders to control for.",
    "Controlling for a confounder (via stratification or partial correlation) can dramatically "
    "change the observed relationship. If controlling for season shrinks r from -0.45 to -0.15, "
    "season was doing most of the work.",
])

navigation(
    prev_label="Ch 59: Autoencoder Anomaly Detection",
    prev_page="59_Autoencoder_Anomaly_Detection.py",
    next_label="Ch 61: Natural Experiments",
    next_page="61_Natural_Experiments.py",
)
