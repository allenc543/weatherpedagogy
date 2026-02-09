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
st.set_page_config(page_title="Ch 61: Natural Experiments", layout="wide")
df = load_data()
fdf = sidebar_filters(df)

chapter_header(61, "Natural Experiments", part="XV")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "Natural Experiments",
    "A <b>natural experiment</b> occurs when some external event or condition creates "
    "variation that mimics a randomised experiment. Unlike a lab experiment, the researcher "
    "does not control the treatment -- nature (or circumstances) does.<br><br>"
    "In weather data, natural experiments arise from:<br>"
    "- Geographic differences (coastal vs inland cities)<br>"
    "- Seasonal transitions (treatment = season change)<br>"
    "- Extreme events (cold front arrival as a 'treatment')",
)

st.markdown("### Two Key Methods")

col1, col2 = st.columns(2)

with col1:
    concept_box(
        "Difference-in-Differences (DiD)",
        "DiD compares the change in outcome over time between a <b>treatment group</b> "
        "and a <b>control group</b>.<br><br>"
        "DiD estimate = (Treatment_after - Treatment_before) - (Control_after - Control_before)<br><br>"
        "The key assumption is <b>parallel trends</b>: without the treatment, both groups "
        "would have followed the same trend.",
    )

with col2:
    concept_box(
        "Granger Causality",
        "Variable X <b>Granger-causes</b> Y if past values of X help predict Y beyond "
        "what past values of Y alone can predict.<br><br>"
        "This is <em>predictive</em> causality, not true causality. But it is useful "
        "for identifying temporal lead-lag relationships.<br><br>"
        "Test: compare an autoregressive model of Y with and without lagged X values.",
    )

formula_box(
    "Difference-in-Differences Estimator",
    r"\hat{\delta}_{DiD} = (\bar{Y}_{T,\text{after}} - \bar{Y}_{T,\text{before}}) "
    r"- (\bar{Y}_{C,\text{after}} - \bar{Y}_{C,\text{before}})",
    "T = treatment group, C = control group. delta_DiD is the causal effect estimate.",
)

formula_box(
    "Granger Causality Test",
    r"Y_t = \alpha + \sum_{i=1}^{p} \beta_i Y_{t-i} + \sum_{i=1}^{p} \gamma_i X_{t-i} + \epsilon_t",
    "Test H0: all gamma_i = 0 (X does not Granger-cause Y). Reject if F-test is significant.",
)

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Granger Causality Test
# ---------------------------------------------------------------------------
st.subheader("Interactive: Does Pressure Granger-Cause Wind Speed?")

st.markdown(
    "Physically, wind is driven by **pressure gradients**. If pressure changes lead to "
    "wind speed changes, we should see pressure Granger-causing wind. Let us test this."
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
display_pw = gc_results_pw.copy()
display_pw["F-statistic"] = display_pw["F-statistic"].map(lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A")
display_pw["p-value"] = display_pw["p-value"].map(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
st.dataframe(display_pw, use_container_width=True, hide_index=True)

n_sig = gc_results_pw["Significant (p<0.05)"].sum()
if n_sig > 0:
    st.success(
        f"Pressure Granger-causes wind speed at {n_sig}/{len(gc_results_pw)} lags tested. "
        "This aligns with meteorological physics: pressure gradients drive wind."
    )
else:
    st.info(
        "Pressure does not Granger-cause wind at the tested lags. "
        "Try a different city or more lags."
    )

if test_reverse:
    st.markdown("#### Wind Speed --> Pressure Results")
    n_sig_rev = gc_results_wp["Significant (p<0.05)"].sum()
    if n_sig_rev > 0:
        st.warning(
            f"Wind also Granger-causes pressure at {n_sig_rev} lags. "
            "This may reflect feedback or shared dynamics, not direct causation."
        )
    else:
        st.success(
            "Wind does NOT Granger-cause pressure. "
            "This asymmetry supports the causal direction: pressure drives wind."
        )

insight_box(
    "Granger causality tests predictive priority, not true causation. "
    "However, when the result aligns with physical theory (pressure gradients drive wind), "
    "it strengthens our causal argument."
)

st.divider()

# ---------------------------------------------------------------------------
# 3. Difference-in-Differences: Coastal vs Inland Temperature Response
# ---------------------------------------------------------------------------
st.subheader("Difference-in-Differences: Summer Effect on Coastal vs Inland Cities")

concept_box(
    "DiD Applied to Weather",
    "We treat the arrival of <b>summer</b> as a 'treatment' and compare how coastal "
    "cities (moderated by the ocean) and inland cities respond differently.<br><br>"
    "<b>Treatment group</b>: Inland cities (expected to heat up more)<br>"
    "<b>Control group</b>: Coastal cities (ocean moderates temperature)<br>"
    "<b>Before</b>: Spring (March-May)<br>"
    "<b>After</b>: Summer (June-August)<br><br>"
    "The DiD estimate captures the <em>differential</em> summer warming between inland "
    "and coastal cities.",
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
""")

warning_box(
    "The DiD estimate is only valid if the parallel trends assumption holds: "
    "without the 'treatment', both groups would have changed by the same amount. "
    "With weather data, geographic differences may violate this assumption."
)

st.divider()

# ---------------------------------------------------------------------------
# 4. Parallel Trends Check
# ---------------------------------------------------------------------------
st.subheader("Parallel Trends Check")

st.markdown(
    "For DiD to be valid, the treatment and control groups should follow **parallel trends** "
    "before the treatment. Let us check this."
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
    "For the DiD estimate to be reliable, the two lines should be roughly parallel "
    "in the 'before' period. If they diverge before treatment, the parallel trends "
    "assumption is violated and the DiD estimate may be biased."
)

st.divider()

# ---------------------------------------------------------------------------
# 5. Additional Granger Tests
# ---------------------------------------------------------------------------
st.subheader("Explore More Granger Causality Relationships")

st.markdown(
    "Test whether any weather variable Granger-causes another."
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
            f"at {n_sig2}/{len(gc2_results)} lags."
        )
    else:
        st.info(
            f"No evidence that {FEATURE_LABELS[gc_x_var]} Granger-causes "
            f"{FEATURE_LABELS[gc_y_var]} at these lags."
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
    "Granger causality tests whether:",
    [
        "X truly causes Y in a philosophical sense",
        "X and Y are correlated",
        "Past values of X help predict Y beyond what past Y alone can predict",
        "X and Y have the same distribution",
    ],
    correct_idx=2,
    explanation="Granger causality is about predictive improvement, not true causation. "
                "X Granger-causes Y if adding past X improves Y forecasts.",
    key="ch61_quiz1",
)

quiz(
    "The key assumption of difference-in-differences is:",
    [
        "Treatment and control groups have the same mean",
        "The treatment is randomly assigned",
        "Parallel trends: both groups would have changed similarly without treatment",
        "The sample size is equal in both groups",
    ],
    correct_idx=2,
    explanation="DiD requires that the treatment and control groups would have followed "
                "parallel paths absent the treatment. This cannot be directly tested.",
    key="ch61_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 8. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Natural experiments exploit external variation to study causal effects without randomization.",
    "Granger causality tests temporal predictive relationships, not true causation.",
    "Pressure Granger-causing wind aligns with meteorological physics (pressure gradients drive wind).",
    "Difference-in-differences compares changes between treatment and control groups.",
    "The parallel trends assumption is crucial for DiD validity and should be checked.",
])

navigation(
    prev_label="Ch 60: Correlation vs Causation",
    prev_page="60_Correlation_vs_Causation.py",
    next_label="Ch 62: Capstone Project",
    next_page="62_Capstone_Project.py",
)
