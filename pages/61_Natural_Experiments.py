"""Chapter 61: Natural Experiments — Difference-in-differences, Granger causality."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, line_chart
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)
from utils.constants import CITY_LIST, CITY_COLORS, FEATURE_COLS, FEATURE_LABELS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
df = load_data()
fdf = sidebar_filters(df)

chapter_header(61, "Natural Experiments", part="XV")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
st.markdown(
    "In the last chapter, we established that correlation does not imply causation. "
    "Season confounds temperature and humidity, city confounds many relationships, and "
    "Simpson's paradox lurks everywhere. But here is the frustrating follow-up question: "
    "if correlation is not enough, how do we *ever* establish causation?"
)
st.markdown(
    "The gold standard is a randomized experiment: randomly assign treatment, measure "
    "outcomes, compare. But we cannot randomize the weather. We cannot flip a coin and "
    "decide 'Dallas gets a high-pressure system this week, Houston does not.' Nature "
    "does not take requests."
)
st.markdown(
    "What we *can* do is find situations where nature itself created something resembling "
    "an experiment. These are called **natural experiments**, and weather data is full of them."
)

concept_box(
    "Natural Experiments in Weather Data",
    "A <b>natural experiment</b> occurs when some external event or condition creates "
    "variation that mimics what a randomized experiment would produce. In our dataset, "
    "here are three concrete examples:<br><br>"
    "<b>1. Coastal vs inland cities</b>: Houston and LA sit on or near the coast; Dallas "
    "and Austin are inland. The ocean moderates temperature swings (nature's treatment "
    "assignment). When summer arrives, we can compare how much more inland cities heat up "
    "relative to coastal ones -- that differential tells us the causal effect of ocean "
    "proximity on temperature response.<br><br>"
    "<b>2. Pressure changes preceding wind</b>: When a cold front approaches Dallas, "
    "surface pressure drops before wind speed increases. This temporal ordering gives us "
    "a quasi-experiment: 'did yesterday's pressure change predict today's wind change, "
    "beyond what yesterday's wind alone could predict?' If yes, that is evidence for a "
    "causal direction (pressure drives wind, not the reverse).<br><br>"
    "<b>3. Seasonal transitions as treatment</b>: The arrival of summer is a 'treatment' "
    "that hits all cities, but affects them differently. We can measure the differential "
    "response between inland and coastal cities to estimate the causal effect of geography "
    "on seasonal temperature sensitivity.",
)

st.markdown("### Two Key Methods")
st.markdown(
    "We will use two tools designed for exactly these situations. Both try to extract "
    "causal information from observational data -- something that regular correlation "
    "analysis cannot do."
)

col1, col2 = st.columns(2)

with col1:
    concept_box(
        "Difference-in-Differences (DiD)",
        "Here is the idea in weather terms. Dallas (inland) and Houston (coastal) both "
        "experience seasons. In spring, Dallas averages 20 degrees C and Houston averages 22. "
        "In summer, Dallas jumps to 33 and Houston to 30.<br><br>"
        "Dallas warmed by 33 - 20 = <b>13 degrees</b>. Houston warmed by 30 - 22 = <b>8 degrees</b>. "
        "The difference-in-differences is 13 - 8 = <b>5 degrees</b>.<br><br>"
        "That 5-degree gap is our estimate of the <em>causal effect</em> of being inland vs "
        "coastal on summer warming. Both cities experienced the same season change, but Dallas "
        "warmed 5 degrees more because it lacks ocean moderation.<br><br>"
        "The key assumption: <b>parallel trends</b>. Without the 'treatment' (summer), both "
        "cities would have changed temperature by the same amount. If Dallas was already "
        "trending warmer for other reasons, our estimate is biased.",
    )

with col2:
    concept_box(
        "Granger Causality",
        "Here is a different question: does yesterday's pressure in Dallas help predict "
        "today's wind speed, <em>beyond</em> what yesterday's wind speed alone can predict?<br><br>"
        "If yes, we say pressure <b>Granger-causes</b> wind. This is not true causation in the "
        "philosophical sense -- it is <em>predictive</em> causality. But when it aligns with "
        "physical theory (pressure gradients drive wind, per the fundamental equations of "
        "meteorology), it strengthens our causal argument considerably.<br><br>"
        "The test works by comparing two models:<br>"
        "- <b>Restricted</b>: predict wind from its own past values only<br>"
        "- <b>Unrestricted</b>: predict wind from its own past + pressure's past<br>"
        "If the unrestricted model is significantly better, pressure adds predictive "
        "information beyond the wind's own history.",
    )

formula_box(
    "Difference-in-Differences Estimator",
    r"\underbrace{\hat{\delta}_{DiD}}_{\text{causal effect estimate}} = (\underbrace{\bar{Y}_{T,\text{after}}}_{\text{treated, after}} - \underbrace{\bar{Y}_{T,\text{before}}}_{\text{treated, before}}) "
    r"- (\underbrace{\bar{Y}_{C,\text{after}}}_{\text{control, after}} - \underbrace{\bar{Y}_{C,\text{before}}}_{\text{control, before}})",
    "T = treatment group (e.g., inland cities), C = control group (e.g., coastal cities). "
    "For our Dallas-vs-Houston summer example: (33 - 20) - (30 - 22) = 13 - 8 = 5 degrees C.",
)

formula_box(
    "Granger Causality Test",
    r"\underbrace{Y_t}_{\text{wind today}} = \underbrace{\alpha}_{\text{intercept}} + \sum_{i=1}^{\underbrace{p}_{\text{lag order}}} \underbrace{\beta_i}_{\text{own-lag weights}} \underbrace{Y_{t-i}}_{\text{past wind}} + \sum_{i=1}^{p} \underbrace{\gamma_i}_{\text{cross-lag weights}} \underbrace{X_{t-i}}_{\text{past pressure}} + \underbrace{\epsilon_t}_{\text{noise}}",
    "Test H0: all gamma_i = 0 (pressure's past does not help predict wind beyond wind's own past). "
    "Reject if F-test is significant -- meaning pressure contains information about future wind "
    "that wind's own history does not.",
)

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Granger Causality Test
# ---------------------------------------------------------------------------
st.subheader("Interactive: Does Pressure Granger-Cause Wind Speed?")

st.markdown(
    "This is one of the cleanest causal hypotheses in meteorology. Wind is driven by "
    "pressure gradients -- differences in atmospheric pressure cause air to move from high "
    "to low pressure. If this physical mechanism is real (spoiler: it is), then past pressure "
    "values should help predict future wind speed."
)
st.markdown(
    "But here is what makes this interesting: does the reverse hold? Does wind Granger-cause "
    "pressure? Physically, wind should NOT drive pressure (wind is a *response* to pressure "
    "gradients, not a cause of them). If our test shows pressure Granger-causes wind but NOT "
    "vice versa, that asymmetry strongly supports the causal direction from pressure to wind."
)
st.markdown(
    "**What the controls do**: Select a city and set the maximum number of lags (how many "
    "days of history to consider). More lags means 'does pressure from up to N days ago "
    "predict today's wind?' The p-value plot shows whether each lag is statistically "
    "significant -- below the dashed line (p < 0.05) means 'yes, this lag helps predict.'"
)

col_ctrl, col_viz = st.columns([1, 2])

with col_ctrl:
    gc_city = st.selectbox("City", CITY_LIST, key="gc_city")
    max_lags = st.slider("Maximum lags (hours)", 1, 48, 12, key="gc_lags")
    st.markdown("---")
    st.markdown("**Also test the reverse:**")
    test_reverse = st.checkbox("Test wind --> pressure", value=True, key="gc_reverse")

city_ts = fdf[fdf["city"] == gc_city].sort_values("datetime").copy()
if len(city_ts) < max_lags + 50:
    st.warning("Not enough data for this analysis.")
    st.stop()

# Prepare time series (use daily averages for cleaner signal)
daily = city_ts.groupby("date").agg({
    "surface_pressure_hpa": "mean",
    "wind_speed_kmh": "mean",
    "temperature_c": "mean",
}).dropna()

# Simple Granger causality implementation
@st.cache_data
def granger_test(y_series, x_series, max_lag):
    """Manual Granger causality test using F-test comparing restricted vs unrestricted model."""
    from numpy.linalg import lstsq

    y = y_series.values
    x = x_series.values
    n = len(y)

    results = []
    for lag in range(1, max_lag + 1):
        if lag >= n - 10:
            break

        # Prepare lagged matrices
        Y = y[lag:]
        Y_lags = np.column_stack([y[lag - i - 1:n - i - 1] for i in range(lag)])
        X_lags = np.column_stack([x[lag - i - 1:n - i - 1] for i in range(lag)])
        T = len(Y)

        # Restricted model: Y ~ Y_lags only
        A_r = np.column_stack([np.ones(T), Y_lags])
        b_r, _, _, _ = lstsq(A_r, Y, rcond=None)
        resid_r = Y - A_r @ b_r
        rss_r = np.sum(resid_r ** 2)

        # Unrestricted model: Y ~ Y_lags + X_lags
        A_u = np.column_stack([np.ones(T), Y_lags, X_lags])
        b_u, _, _, _ = lstsq(A_u, Y, rcond=None)
        resid_u = Y - A_u @ b_u
        rss_u = np.sum(resid_u ** 2)

        # F-test
        df_num = lag  # number of restrictions
        df_den = T - 2 * lag - 1

        if df_den > 0 and rss_u > 0:
            f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
            p_value = 1 - stats.f.cdf(f_stat, df_num, df_den)
        else:
            f_stat = np.nan
            p_value = np.nan

        results.append({
            "Lag": lag,
            "F-statistic": f_stat,
            "p-value": p_value,
            "Significant (p<0.05)": p_value < 0.05 if not np.isnan(p_value) else False,
        })

    return pd.DataFrame(results)

# Test: pressure --> wind
gc_results_pw = granger_test(daily["wind_speed_kmh"], daily["surface_pressure_hpa"], max_lags)

with col_viz:
    fig_gc = go.Figure()
    fig_gc.add_trace(go.Scatter(
        x=gc_results_pw["Lag"], y=gc_results_pw["p-value"],
        mode="lines+markers",
        line=dict(color="#E63946", width=2),
        name="Pressure --> Wind",
    ))

    if test_reverse:
        gc_results_wp = granger_test(daily["surface_pressure_hpa"], daily["wind_speed_kmh"], max_lags)
        fig_gc.add_trace(go.Scatter(
            x=gc_results_wp["Lag"], y=gc_results_wp["p-value"],
            mode="lines+markers",
            line=dict(color="#2A9D8F", width=2),
            name="Wind --> Pressure",
        ))

    fig_gc.add_hline(y=0.05, line_dash="dash", line_color="black",
                     annotation_text="p = 0.05 threshold")
    fig_gc.update_layout(
        xaxis_title="Lag (days)", yaxis_title="p-value",
        yaxis_type="log",
    )
    apply_common_layout(fig_gc, title=f"Granger Causality Test Results ({gc_city})", height=450)
    st.plotly_chart(fig_gc, use_container_width=True)

st.markdown("#### Pressure --> Wind Speed Results")
st.markdown(
    "Each row tests: 'Does pressure history up to N days ago significantly improve "
    "wind speed prediction?' A p-value below 0.05 means yes."
)
display_pw = gc_results_pw.copy()
display_pw["F-statistic"] = display_pw["F-statistic"].map(lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A")
display_pw["p-value"] = display_pw["p-value"].map(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
st.dataframe(display_pw, use_container_width=True, hide_index=True)

n_sig = gc_results_pw["Significant (p<0.05)"].sum()
if n_sig > 0:
    st.success(
        f"Pressure Granger-causes wind speed at **{n_sig}/{len(gc_results_pw)}** lags tested. "
        "This aligns perfectly with meteorological physics: pressure gradients are the "
        "fundamental driver of wind. Past pressure changes in the region contain information "
        "about future wind that the wind's own history does not."
    )
else:
    st.info(
        "Pressure does not Granger-cause wind at the tested lags for this city. "
        "This can happen with daily averages (the effect may be sub-daily) or in cities "
        "with complex terrain. Try a different city or increase the lag range."
    )

if test_reverse:
    st.markdown("#### Wind Speed --> Pressure Results")
    n_sig_rev = gc_results_wp["Significant (p<0.05)"].sum()
    if n_sig_rev > 0:
        st.warning(
            f"Wind also Granger-causes pressure at **{n_sig_rev}** lags. This may reflect "
            "feedback loops in atmospheric dynamics (wind transports air masses that change "
            "local pressure) or shared responses to larger weather systems. It does not mean "
            "wind *directly* causes pressure changes in the way pressure gradients cause wind."
        )
    else:
        st.success(
            "Wind does NOT Granger-cause pressure -- exactly what physics predicts. The "
            "asymmetry is telling: pressure helps predict wind (causal direction: pressure "
            "--> wind), but wind does not help predict pressure (wind is a response, not a "
            "cause). This one-directional Granger causality aligns with the physical mechanism."
        )

insight_box(
    "Granger causality tests temporal predictive relationships, not true causation in a "
    "philosophical sense. But when the result aligns with well-established physical theory "
    "(pressure gradients drive wind), AND the reverse test fails (wind does not predict "
    "pressure), the combined evidence is much stronger than either test alone. The asymmetry "
    "is the key: if both directions were significant, it would suggest a common driver rather "
    "than a direct causal link."
)

st.divider()

# ---------------------------------------------------------------------------
# 3. Difference-in-Differences: Coastal vs Inland Temperature Response
# ---------------------------------------------------------------------------
st.subheader("Difference-in-Differences: Summer Effect on Coastal vs Inland Cities")

st.markdown(
    "Now for the DiD analysis. Here is the setup in concrete terms."
)
st.markdown(
    "**The 'treatment'**: Summer arrives. Every city gets hotter, but the question is: "
    "how much MORE do inland cities heat up compared to coastal cities? The ocean absorbs "
    "heat and moderates temperature swings -- so coastal cities should warm less. The "
    "difference-in-differences estimates this *causal effect of geography* on seasonal "
    "temperature response."
)

concept_box(
    "DiD Applied to Our Weather Data",
    "Think of it as two comparisons stacked on top of each other:<br><br>"
    "<b>Treatment group</b>: Inland cities (Dallas, Austin, San Antonio) -- expected to "
    "heat up more in summer because they lack ocean moderation.<br>"
    "<b>Control group</b>: Coastal cities (Houston, Los Angeles) -- the ocean acts as a "
    "thermal buffer, moderating temperature swings.<br>"
    "<b>Before period</b>: Spring (March-May) -- our pre-treatment baseline.<br>"
    "<b>After period</b>: Summer (June-August) -- the 'treatment' is applied.<br><br>"
    "Dallas goes from 20 degrees C in spring to 33 degrees C in summer (+13). "
    "Houston goes from 22 degrees C in spring to 30 degrees C in summer (+8). "
    "DiD estimate = 13 - 8 = <b>5 degrees C</b>. That 5-degree gap is the estimated "
    "causal effect of being inland rather than coastal on summer warming.<br><br>"
    "The dotted 'counterfactual' line on the chart shows where Dallas <em>would have "
    "been</em> if it had warmed at the same rate as Houston. The gap between the actual "
    "Dallas temperature and the counterfactual is the DiD estimate.",
)

# Select treatment and control cities
col_did1, col_did2 = st.columns(2)

with col_did1:
    from utils.constants import COASTAL_CITIES, INLAND_CITIES

    available_coastal = [c for c in COASTAL_CITIES if c in fdf["city"].unique()]
    available_inland = [c for c in INLAND_CITIES if c in fdf["city"].unique()]

    if not available_coastal:
        available_coastal = CITY_LIST[:1]
    if not available_inland:
        available_inland = CITY_LIST[1:2]

    treatment_cities = st.multiselect(
        "Treatment (inland) cities", CITY_LIST,
        default=available_inland[:2] if len(available_inland) >= 2 else available_inland,
        key="did_treat",
    )
    control_cities = st.multiselect(
        "Control (coastal/reference) cities", CITY_LIST,
        default=available_coastal[:1],
        key="did_ctrl",
    )

with col_did2:
    before_months = st.multiselect(
        "Before period (months)", list(range(1, 13)),
        default=[3, 4, 5],
        format_func=lambda m: pd.Timestamp(2024, m, 1).month_name(),
        key="did_before",
    )
    after_months = st.multiselect(
        "After period (months)", list(range(1, 13)),
        default=[6, 7, 8],
        format_func=lambda m: pd.Timestamp(2024, m, 1).month_name(),
        key="did_after",
    )

if not treatment_cities or not control_cities or not before_months or not after_months:
    st.warning("Please select cities and time periods.")
    st.stop()

did_feature = st.selectbox(
    "Outcome variable", FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    key="did_feature",
)

# Compute DiD
treat_before = fdf[
    fdf["city"].isin(treatment_cities) & fdf["month"].isin(before_months)
][did_feature].mean()

treat_after = fdf[
    fdf["city"].isin(treatment_cities) & fdf["month"].isin(after_months)
][did_feature].mean()

ctrl_before = fdf[
    fdf["city"].isin(control_cities) & fdf["month"].isin(before_months)
][did_feature].mean()

ctrl_after = fdf[
    fdf["city"].isin(control_cities) & fdf["month"].isin(after_months)
][did_feature].mean()

did_estimate = (treat_after - treat_before) - (ctrl_after - ctrl_before)

# Visualise
fig_did = go.Figure()

# Treatment group
fig_did.add_trace(go.Scatter(
    x=["Before", "After"],
    y=[treat_before, treat_after],
    mode="lines+markers",
    line=dict(color="#E63946", width=3),
    marker=dict(size=12),
    name=f"Treatment ({', '.join(treatment_cities)})",
))

# Control group
fig_did.add_trace(go.Scatter(
    x=["Before", "After"],
    y=[ctrl_before, ctrl_after],
    mode="lines+markers",
    line=dict(color="#2A9D8F", width=3),
    marker=dict(size=12),
    name=f"Control ({', '.join(control_cities)})",
))

# Counterfactual (parallel trend for treatment)
counterfactual_after = treat_before + (ctrl_after - ctrl_before)
fig_did.add_trace(go.Scatter(
    x=["Before", "After"],
    y=[treat_before, counterfactual_after],
    mode="lines",
    line=dict(color="#E63946", width=2, dash="dot"),
    name="Counterfactual (parallel trend)",
))

# DiD annotation
fig_did.add_annotation(
    x="After",
    y=(treat_after + counterfactual_after) / 2,
    text=f"DiD = {did_estimate:.2f}",
    showarrow=True,
    arrowhead=2,
    font=dict(size=14, color="#264653"),
)

fig_did.update_layout(
    yaxis_title=FEATURE_LABELS.get(did_feature, did_feature),
    xaxis_title="Period",
)
apply_common_layout(fig_did, title="Difference-in-Differences Estimate", height=450)
st.plotly_chart(fig_did, use_container_width=True)

# DiD summary table
did_table = pd.DataFrame({
    "": ["Treatment", "Control", "Difference"],
    "Before": [f"{treat_before:.2f}", f"{ctrl_before:.2f}",
               f"{treat_before - ctrl_before:.2f}"],
    "After": [f"{treat_after:.2f}", f"{ctrl_after:.2f}",
              f"{treat_after - ctrl_after:.2f}"],
    "Change": [f"{treat_after - treat_before:.2f}",
               f"{ctrl_after - ctrl_before:.2f}",
               f"{did_estimate:.2f} (DiD)"],
})
st.dataframe(did_table, use_container_width=True, hide_index=True)

st.markdown(f"""
**Interpretation**: The DiD estimate is **{did_estimate:.2f} {FEATURE_LABELS.get(did_feature, did_feature).split('(')[-1].replace(')', '') if '(' in FEATURE_LABELS.get(did_feature, did_feature) else ''}**.

This means that going from the before period to the after period, the treatment cities
experienced a **{abs(did_estimate):.2f}** unit {'greater increase' if did_estimate > 0 else 'greater decrease'}
in {FEATURE_LABELS.get(did_feature, did_feature).lower()} compared to the control cities.
For temperature, this is the extra warming (or reduced warming) that inland cities experience
relative to coastal cities -- our estimate of the causal effect of geography on seasonal response.
""")

warning_box(
    "The DiD estimate is only valid if the parallel trends assumption holds: without the "
    "'treatment' (summer), both groups would have changed by the same amount. For weather, "
    "this is plausible for season-driven changes but can break down if one city group has "
    "a different underlying trend (e.g., urban heat island effects growing faster in one group). "
    "Check the parallel trends plot below to assess this assumption."
)

st.divider()

# ---------------------------------------------------------------------------
# 4. Parallel Trends Check
# ---------------------------------------------------------------------------
st.subheader("Parallel Trends Check")

st.markdown(
    "DiD requires that the treatment and control groups would have followed **parallel "
    "trends** without the treatment. We cannot test this directly (we cannot observe the "
    "counterfactual), but we CAN check whether the trends were parallel in the *pre-treatment* "
    "period. If the two lines move together before the treatment, it is more plausible that "
    "they would have continued to move together without it."
)
st.markdown(
    "Look at the chart below. In the months you selected as 'before,' are the treatment "
    "and control lines roughly parallel? If they are already diverging before the 'after' "
    "period, the parallel trends assumption is suspect and the DiD estimate may be biased."
)

# Monthly averages for both groups
monthly_treat = fdf[fdf["city"].isin(treatment_cities)].groupby("month")[did_feature].mean()
monthly_ctrl = fdf[fdf["city"].isin(control_cities)].groupby("month")[did_feature].mean()

month_names = {i: pd.Timestamp(2024, i, 1).month_name()[:3] for i in range(1, 13)}

fig_pt = go.Figure()
fig_pt.add_trace(go.Scatter(
    x=[month_names.get(m, str(m)) for m in monthly_treat.index],
    y=monthly_treat.values,
    mode="lines+markers",
    line=dict(color="#E63946", width=2),
    name=f"Treatment ({', '.join(treatment_cities)})",
))
fig_pt.add_trace(go.Scatter(
    x=[month_names.get(m, str(m)) for m in monthly_ctrl.index],
    y=monthly_ctrl.values,
    mode="lines+markers",
    line=dict(color="#2A9D8F", width=2),
    name=f"Control ({', '.join(control_cities)})",
))

# Shade before/after periods
for m in before_months:
    if m in monthly_treat.index:
        fig_pt.add_vrect(
            x0=month_names.get(m, str(m)), x1=month_names.get(m, str(m)),
            fillcolor="rgba(42,157,143,0.1)", line_width=0,
        )
for m in after_months:
    if m in monthly_treat.index:
        fig_pt.add_vrect(
            x0=month_names.get(m, str(m)), x1=month_names.get(m, str(m)),
            fillcolor="rgba(230,57,70,0.1)", line_width=0,
        )

fig_pt.update_layout(
    xaxis_title="Month",
    yaxis_title=FEATURE_LABELS.get(did_feature, did_feature),
)
apply_common_layout(fig_pt, title="Monthly Trends: Treatment vs Control", height=400)
st.plotly_chart(fig_pt, use_container_width=True)

insight_box(
    "For the DiD estimate to be reliable, the two lines should be roughly parallel in the "
    "'before' period (the green-shaded months). If they are -- if inland and coastal cities "
    "were warming at the same rate during spring -- then it is plausible that any divergence "
    "in summer is caused by the differential effect of ocean proximity. If the lines were "
    "already diverging in spring, the parallel trends assumption is violated, and the DiD "
    "estimate is capturing pre-existing differences rather than a causal effect."
)

st.divider()

# ---------------------------------------------------------------------------
# 5. Additional Granger Tests
# ---------------------------------------------------------------------------
st.subheader("Explore More Granger Causality Relationships")

st.markdown(
    "The pressure-wind relationship is the clearest test case, but you can test any "
    "pair of weather variables. Some interesting ones to try:"
)
st.markdown(
    "- **Temperature --> Humidity**: Does today's temperature predict tomorrow's humidity "
    "change? (Physically plausible through evaporation.)\n"
    "- **Wind --> Temperature**: Do high winds predict temperature drops? (Cold fronts bring "
    "wind before temperature changes.)\n"
    "- **Pressure --> Temperature**: Do pressure changes predict temperature changes? "
    "(Dropping pressure often precedes cold front arrival.)"
)

gc_x_var = st.selectbox(
    "Potential cause (X)", FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    index=3,  # pressure
    key="gc2_x",
)
gc_y_var = st.selectbox(
    "Potential effect (Y)", FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    index=2,  # wind
    key="gc2_y",
)
gc2_city = st.selectbox("City", CITY_LIST, key="gc2_city")
gc2_lags = st.slider("Max lags", 1, 30, 7, key="gc2_lags")

daily2 = fdf[fdf["city"] == gc2_city].sort_values("datetime").groupby("date").agg({
    c: "mean" for c in FEATURE_COLS
}).dropna()

if len(daily2) > gc2_lags + 10:
    gc2_results = granger_test(daily2[gc_y_var], daily2[gc_x_var], gc2_lags)

    st.markdown(f"**{FEATURE_LABELS[gc_x_var]} --> {FEATURE_LABELS[gc_y_var]}**")
    disp2 = gc2_results.copy()
    disp2["F-statistic"] = disp2["F-statistic"].map(lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A")
    disp2["p-value"] = disp2["p-value"].map(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
    st.dataframe(disp2, use_container_width=True, hide_index=True)

    n_sig2 = gc2_results["Significant (p<0.05)"].sum()
    if n_sig2 > 0:
        st.success(
            f"{FEATURE_LABELS[gc_x_var]} Granger-causes {FEATURE_LABELS[gc_y_var]} "
            f"at **{n_sig2}/{len(gc2_results)}** lags. Past values of "
            f"{FEATURE_LABELS[gc_x_var].lower()} contain information about future "
            f"{FEATURE_LABELS[gc_y_var].lower()} beyond what its own past provides."
        )
    else:
        st.info(
            f"No evidence that {FEATURE_LABELS[gc_x_var]} Granger-causes "
            f"{FEATURE_LABELS[gc_y_var]} at these lags. The past values of "
            f"{FEATURE_LABELS[gc_x_var].lower()} do not add predictive value beyond "
            f"what {FEATURE_LABELS[gc_y_var].lower()}'s own history provides."
        )
else:
    st.warning("Not enough daily data for this test.")

st.divider()

# ---------------------------------------------------------------------------
# 6. Code Example
# ---------------------------------------------------------------------------
code_example("""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

# --- Granger Causality ---
# Using statsmodels (recommended for production)
daily = city_df.resample('D', on='datetime').mean()
test_data = daily[['surface_pressure_hpa', 'wind_speed_kmh']].dropna()
result = grangercausalitytests(test_data, maxlag=12, verbose=False)

for lag, res in result.items():
    f_stat = res[0]['ssr_ftest'][0]
    p_val = res[0]['ssr_ftest'][1]
    print(f"Lag {lag}: F={f_stat:.2f}, p={p_val:.4f}")

# --- Difference-in-Differences ---
# Treatment: inland cities in summer vs spring
# Control: coastal cities in summer vs spring
treat_before = df[(df['city'].isin(inland)) & (df['season'] == 'Spring')]['temperature_c'].mean()
treat_after = df[(df['city'].isin(inland)) & (df['season'] == 'Summer')]['temperature_c'].mean()
ctrl_before = df[(df['city'].isin(coastal)) & (df['season'] == 'Spring')]['temperature_c'].mean()
ctrl_after = df[(df['city'].isin(coastal)) & (df['season'] == 'Summer')]['temperature_c'].mean()

did = (treat_after - treat_before) - (ctrl_after - ctrl_before)
print(f"DiD estimate: {did:.2f} degrees C")
""")

st.divider()

# ---------------------------------------------------------------------------
# 7. Quiz
# ---------------------------------------------------------------------------
quiz(
    "You run Granger causality in Dallas and find that pressure Granger-causes wind (p = 0.002) "
    "but wind does NOT Granger-cause pressure (p = 0.45). What does this asymmetry tell you?",
    [
        "There is no relationship between pressure and wind",
        "The causal direction runs from pressure to wind -- consistent with the physical "
        "mechanism that pressure gradients drive wind, not the reverse",
        "Wind causes pressure changes but only at longer lags",
        "Both variables are driven by the same confounder",
    ],
    correct_idx=1,
    explanation="The asymmetry is the key finding. Pressure's past helps predict wind's future "
                "(p = 0.002, highly significant), but wind's past does NOT help predict pressure's "
                "future (p = 0.45, not even close to significant). This one-directional predictive "
                "relationship aligns perfectly with meteorological physics: pressure differences "
                "(gradients) drive air movement (wind), but wind does not create the pressure "
                "differences that drive it. Granger causality alone is not proof of true causation, "
                "but when it matches the known physical mechanism AND shows the expected asymmetry, "
                "the combined evidence is compelling.",
    key="ch61_quiz1",
)

quiz(
    "In a DiD analysis, Dallas (inland) warms by 13 degrees C from spring to summer, while "
    "Houston (coastal) warms by 8 degrees C. The DiD estimate is 5 degrees C. What does this "
    "5 degrees represent?",
    [
        "The average temperature difference between Dallas and Houston",
        "The total warming in Dallas during summer",
        "The estimated causal effect of being inland vs coastal on seasonal warming -- "
        "the extra warming Dallas experiences because it lacks ocean moderation",
        "The measurement error between the two cities",
    ],
    correct_idx=2,
    explanation="The 5-degree DiD estimate captures the *differential* warming: how much more "
                "Dallas heats up relative to Houston. Both cities experienced summer (the common "
                "'treatment'), but Dallas warmed 5 degrees more because it lacks the ocean's "
                "thermal buffer. This is our estimate of the causal effect of geography (inland "
                "vs coastal) on temperature response to seasonal change. The key assumption is "
                "that without any geographical difference, both cities would have warmed by the "
                "same amount -- the parallel trends assumption.",
    key="ch61_quiz2",
)

quiz(
    "You check parallel trends for your DiD analysis and find that inland cities were already "
    "warming faster than coastal cities during the 'before' period (spring). What does this "
    "mean for your DiD estimate?",
    [
        "The DiD estimate is still perfectly valid",
        "The parallel trends assumption is violated, so the DiD estimate is likely biased -- "
        "it attributes pre-existing differential warming to the 'treatment' (summer)",
        "You need more data to determine if the trend is real",
        "You should switch to Granger causality instead",
    ],
    correct_idx=1,
    explanation="Parallel trends is the core assumption of DiD. If inland cities were already "
                "warming faster than coastal cities during spring (before the 'treatment' of "
                "summer), then some of the differential warming you attribute to summer was "
                "actually a continuation of a pre-existing trend. Your DiD estimate is biased "
                "upward -- it includes both the real summer effect AND the pre-existing divergence. "
                "You might try adjusting the before period, adding more controls, or using a "
                "different estimation strategy (like matching on pre-trends).",
    key="ch61_quiz3",
)

st.divider()

# ---------------------------------------------------------------------------
# 8. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Natural experiments exploit situations where nature creates variation resembling a "
    "controlled experiment. In weather data, geographic differences (coastal vs inland) and "
    "temporal ordering (pressure changes before wind changes) provide natural experiments.",
    "Granger causality tests whether past values of X improve predictions of Y beyond Y's "
    "own history. For pressure and wind in our data, the test often confirms the physical "
    "mechanism: pressure Granger-causes wind, but not vice versa. The asymmetry is the "
    "strongest evidence.",
    "Difference-in-differences compares the change in outcomes between treatment and control "
    "groups. For our cities, the DiD estimate captures how much more inland cities warm in "
    "summer compared to coastal cities -- roughly 3-5 degrees C depending on the cities chosen.",
    "The parallel trends assumption is the Achilles heel of DiD: both groups must have been "
    "trending similarly before the treatment. Always check this visually -- if the lines "
    "are already diverging before treatment, your estimate is compromised.",
    "Neither method proves causation with certainty. Granger causality is about prediction, "
    "not mechanism. DiD requires an untestable assumption. But when the results align with "
    "known physical theory, the combined evidence is much stronger than any single method.",
])

navigation(
    prev_label="Ch 60: Correlation vs Causation",
    prev_page="60_Correlation_vs_Causation.py",
    next_label="Ch 62: Capstone Project",
    next_page="62_Capstone_Project.py",
)
