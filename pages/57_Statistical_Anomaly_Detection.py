"""Chapter 57: Statistical Anomaly Detection — Z-score, IQR, contextual anomalies."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, histogram_chart
from utils.stats_helpers import descriptive_stats
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)
from utils.constants import CITY_LIST, CITY_COLORS, FEATURE_COLS, FEATURE_LABELS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Ch 57: Statistical Anomaly Detection", layout="wide")
df = load_data()
fdf = sidebar_filters(df)

chapter_header(57, "Statistical Anomaly Detection", part="XIV")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "What Is Anomaly Detection?",
    "Anomaly detection is the art of answering the question: 'is this data point weird?' "
    "Which sounds simple until you realize you need a rigorous definition of 'weird.' "
    "Statistical methods handle this by learning what <b>normal</b> looks like from your data, "
    "then flagging points that deviate significantly from that baseline.<br><br>"
    "In weather data, anomalies are the interesting stuff:<br>"
    "- <b>Heat waves</b>: unusually high temperatures for the season<br>"
    "- <b>Cold snaps</b>: the time Dallas hit -15 C, which residents would enthusiastically "
    "confirm was indeed weird<br>"
    "- <b>Storms</b>: extreme wind speeds or sudden pressure drops<br><br>"
    "The tricky part is that 'unusual' depends on context. 35 C in July is Tuesday. "
    "35 C in January is a crisis.",
)

st.markdown("### Two Classical Methods")

col1, col2 = st.columns(2)

with col1:
    concept_box(
        "Z-Score Method",
        "Measures how many standard deviations a point is from the mean. "
        "A point is anomalous if |z| > threshold (typically 2 or 3).<br><br>"
        "The appeal is simplicity: you are literally just asking 'how far is this from average, "
        "in units of typical spread?'<br><br>"
        "<b>Pros</b>: Simple, intuitive, works well when your data is roughly normal.<br>"
        "<b>Cons</b>: The mean and std are themselves influenced by outliers, which is a bit like "
        "asking the suspects to serve on the jury. Also assumes normality.",
    )
    formula_box(
        "Z-Score",
        r"z = \frac{x - \mu}{\sigma}",
        "A z-score of 3 means the point is 3 standard deviations from the mean -- "
        "expected to happen about 0.3% of the time under normality.",
    )

with col2:
    concept_box(
        "IQR Method",
        "Uses the interquartile range (Q3 - Q1) to define bounds. "
        "Anomalies fall below Q1 - k*IQR or above Q3 + k*IQR (typically k=1.5).<br><br>"
        "The key advantage: quartiles do not care about outliers. You could replace the most "
        "extreme value with infinity and the IQR would not change. This makes the method "
        "more robust when your data is contaminated.<br><br>"
        "<b>Pros</b>: Robust to outliers; no normality assumption needed.<br>"
        "<b>Cons</b>: May miss subtle anomalies; the k=1.5 multiplier is essentially arbitrary.",
    )
    formula_box(
        "IQR Bounds",
        r"\text{Lower} = Q_1 - k \cdot IQR, \quad \text{Upper} = Q_3 + k \cdot IQR",
        "With k=1.5, this corresponds roughly to 2.7 sigma for normal data. "
        "John Tukey picked this value in 1977 and nobody has managed to improve on it much since.",
    )

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Anomaly Detection on Time Series
# ---------------------------------------------------------------------------
st.subheader("Interactive: Detect Weather Anomalies")

col_ctrl, col_viz = st.columns([1, 3])

with col_ctrl:
    ad_city = st.selectbox("City", CITY_LIST, key="ad_city")
    ad_feature = st.selectbox(
        "Feature",
        FEATURE_COLS,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
        key="ad_feature",
    )
    ad_method = st.radio("Method", ["Z-Score", "IQR"], key="ad_method")

    if ad_method == "Z-Score":
        ad_threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1, key="ad_zthresh")
    else:
        ad_iqr_k = st.slider("IQR multiplier (k)", 0.5, 4.0, 1.5, 0.1, key="ad_iqrk")

    contextual = st.checkbox("Contextual (per-month)", value=False, key="ad_contextual",
                             help="Compute anomaly bounds separately for each month")

city_data = fdf[fdf["city"] == ad_city].copy().sort_values("datetime")
if len(city_data) < 10:
    st.warning("Not enough data. Adjust filters.")
    st.stop()

feature_col = ad_feature
values = city_data[feature_col].values

# Compute anomaly flags
if contextual:
    # Per-month statistics
    city_data["is_anomaly"] = False
    city_data["lower_bound"] = np.nan
    city_data["upper_bound"] = np.nan

    for month in range(1, 13):
        mask = city_data["month"] == month
        if mask.sum() < 5:
            continue
        month_vals = city_data.loc[mask, feature_col]

        if ad_method == "Z-Score":
            mu = month_vals.mean()
            sigma = month_vals.std()
            z_scores = (month_vals - mu) / (sigma if sigma > 0 else 1)
            city_data.loc[mask, "is_anomaly"] = np.abs(z_scores) > ad_threshold
            city_data.loc[mask, "lower_bound"] = mu - ad_threshold * sigma
            city_data.loc[mask, "upper_bound"] = mu + ad_threshold * sigma
        else:
            q1 = month_vals.quantile(0.25)
            q3 = month_vals.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - ad_iqr_k * iqr
            upper = q3 + ad_iqr_k * iqr
            city_data.loc[mask, "is_anomaly"] = (month_vals < lower) | (month_vals > upper)
            city_data.loc[mask, "lower_bound"] = lower
            city_data.loc[mask, "upper_bound"] = upper
else:
    if ad_method == "Z-Score":
        mu = values.mean()
        sigma = values.std()
        z_scores = (values - mu) / (sigma if sigma > 0 else 1)
        city_data["is_anomaly"] = np.abs(z_scores) > ad_threshold
        city_data["lower_bound"] = mu - ad_threshold * sigma
        city_data["upper_bound"] = mu + ad_threshold * sigma
    else:
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower = q1 - ad_iqr_k * iqr
        upper = q3 + ad_iqr_k * iqr
        city_data["is_anomaly"] = (values < lower) | (values > upper)
        city_data["lower_bound"] = lower
        city_data["upper_bound"] = upper

n_anomalies = city_data["is_anomaly"].sum()
pct_anomalies = n_anomalies / len(city_data) * 100

# Plot
with col_viz:
    fig = go.Figure()

    # Normal points
    normal = city_data[~city_data["is_anomaly"]]
    fig.add_trace(go.Scatter(
        x=normal["datetime"], y=normal[feature_col],
        mode="markers", marker=dict(color=CITY_COLORS.get(ad_city, "#2A9D8F"), size=2, opacity=0.3),
        name="Normal",
    ))

    # Anomalies
    anomalies = city_data[city_data["is_anomaly"]]
    fig.add_trace(go.Scatter(
        x=anomalies["datetime"], y=anomalies[feature_col],
        mode="markers", marker=dict(color="#E63946", size=6, symbol="x"),
        name=f"Anomalies ({n_anomalies})",
    ))

    # Bounds
    fig.add_trace(go.Scatter(
        x=city_data["datetime"], y=city_data["upper_bound"],
        mode="lines", line=dict(color="#F4A261", width=1, dash="dash"),
        name="Upper Bound",
    ))
    fig.add_trace(go.Scatter(
        x=city_data["datetime"], y=city_data["lower_bound"],
        mode="lines", line=dict(color="#F4A261", width=1, dash="dash"),
        name="Lower Bound", fill="tonexty", fillcolor="rgba(244,162,97,0.08)",
    ))

    fig.update_layout(xaxis_title="Date", yaxis_title=FEATURE_LABELS.get(feature_col, feature_col))
    apply_common_layout(
        fig,
        title=f"Anomaly Detection: {FEATURE_LABELS[feature_col]} in {ad_city} ({ad_method})",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

m1, m2, m3 = st.columns(3)
m1.metric("Total Points", f"{len(city_data):,}")
m2.metric("Anomalies Detected", f"{n_anomalies:,}")
m3.metric("Anomaly Rate", f"{pct_anomalies:.2f}%")

if contextual:
    insight_box(
        "Contextual detection uses per-month statistics, which is crucial for weather data. "
        "A 35 deg C reading in January is genuinely alarming; the same reading in July is just "
        "Texas being Texas. Without context, you would either miss winter anomalies or flag "
        "every summer day as unusual."
    )
else:
    insight_box(
        "Global detection uses the overall distribution, which means it treats a July heatwave "
        "and a January cold snap with the same yardstick. This can miss seasonally contextual "
        "anomalies. Try enabling the 'Contextual (per-month)' checkbox to see the difference."
    )

st.divider()

# ---------------------------------------------------------------------------
# 3. Anomaly Distribution Analysis
# ---------------------------------------------------------------------------
st.subheader("Anomaly Distribution Analysis")

if n_anomalies > 0:
    col_a, col_b = st.columns(2)

    with col_a:
        # When do anomalies occur?
        anom_months = anomalies["month_name"].value_counts().reindex(
            ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]
        ).fillna(0)

        fig_month = go.Figure(go.Bar(
            x=anom_months.index, y=anom_months.values,
            marker_color="#E63946",
        ))
        fig_month.update_layout(xaxis_title="Month", yaxis_title="Anomaly Count")
        apply_common_layout(fig_month, title="Anomalies by Month", height=400)
        st.plotly_chart(fig_month, use_container_width=True)

    with col_b:
        # Hour distribution
        anom_hours = anomalies["hour"].value_counts().sort_index()
        fig_hour = go.Figure(go.Bar(
            x=anom_hours.index, y=anom_hours.values,
            marker_color="#7209B7",
        ))
        fig_hour.update_layout(xaxis_title="Hour of Day", yaxis_title="Anomaly Count")
        apply_common_layout(fig_hour, title="Anomalies by Hour", height=400)
        st.plotly_chart(fig_hour, use_container_width=True)

    # Classify anomalies
    st.markdown("#### Anomaly Classification")

    if feature_col == "temperature_c":
        overall_mean = city_data[feature_col].mean()
        heat_waves = anomalies[anomalies[feature_col] > overall_mean]
        cold_snaps = anomalies[anomalies[feature_col] <= overall_mean]
        st.markdown(
            f"- **Heat waves** (above mean): {len(heat_waves)} events\n"
            f"- **Cold snaps** (below mean): {len(cold_snaps)} events"
        )
    elif feature_col == "wind_speed_kmh":
        st.markdown(f"- **Wind storms** (extreme wind): {n_anomalies} events")
    elif feature_col == "surface_pressure_hpa":
        overall_mean = city_data[feature_col].mean()
        low_press = anomalies[anomalies[feature_col] < overall_mean]
        high_press = anomalies[anomalies[feature_col] >= overall_mean]
        st.markdown(
            f"- **Low pressure anomalies** (storm-like): {len(low_press)} events\n"
            f"- **High pressure anomalies**: {len(high_press)} events"
        )

    # Show sample anomalies
    st.markdown("#### Sample Anomalous Readings")
    display_cols = ["datetime", "city", feature_col, "lower_bound", "upper_bound"]
    if feature_col == "temperature_c":
        display_cols += ["relative_humidity_pct", "wind_speed_kmh"]
    st.dataframe(
        anomalies[display_cols].head(20).reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )
else:
    st.info("No anomalies detected with current settings. Try lowering the threshold.")

st.divider()

# ---------------------------------------------------------------------------
# 4. Method Comparison
# ---------------------------------------------------------------------------
st.subheader("Method Comparison: Z-Score vs IQR")

st.markdown(
    "A reasonable question: if both methods detect anomalies, do they agree? "
    "The answer is 'sometimes,' and the disagreements are instructive."
)

comp_city = ad_city
comp_data = fdf[fdf["city"] == comp_city].copy().sort_values("datetime")
comp_vals = comp_data[feature_col].values

# Z-score anomalies (z > 3)
comp_mu = comp_vals.mean()
comp_sigma = comp_vals.std()
comp_z = (comp_vals - comp_mu) / (comp_sigma if comp_sigma > 0 else 1)
z_anomaly = np.abs(comp_z) > 3.0

# IQR anomalies (k = 1.5)
comp_q1 = np.percentile(comp_vals, 25)
comp_q3 = np.percentile(comp_vals, 75)
comp_iqr = comp_q3 - comp_q1
iqr_anomaly = (comp_vals < comp_q1 - 1.5 * comp_iqr) | (comp_vals > comp_q3 + 1.5 * comp_iqr)

both = z_anomaly & iqr_anomaly
z_only = z_anomaly & ~iqr_anomaly
iqr_only = ~z_anomaly & iqr_anomaly

comp_stats = pd.DataFrame({
    "Method": ["Z-Score (|z|>3)", "IQR (k=1.5)", "Both Methods", "Z-Score Only", "IQR Only"],
    "Anomalies": [z_anomaly.sum(), iqr_anomaly.sum(), both.sum(), z_only.sum(), iqr_only.sum()],
    "Percentage": [
        f"{z_anomaly.mean()*100:.2f}%",
        f"{iqr_anomaly.mean()*100:.2f}%",
        f"{both.mean()*100:.2f}%",
        f"{z_only.mean()*100:.2f}%",
        f"{iqr_only.mean()*100:.2f}%",
    ],
})
st.dataframe(comp_stats, use_container_width=True, hide_index=True)

# Distribution with both thresholds
fig_comp = go.Figure()
fig_comp.add_trace(go.Histogram(
    x=comp_vals, nbinsx=80, name="Data",
    marker_color=CITY_COLORS.get(comp_city, "#2A9D8F"), opacity=0.7,
))

# Z-score bounds
fig_comp.add_vline(x=comp_mu - 3 * comp_sigma, line_dash="solid", line_color="#E63946",
                   annotation_text="Z=-3")
fig_comp.add_vline(x=comp_mu + 3 * comp_sigma, line_dash="solid", line_color="#E63946",
                   annotation_text="Z=+3")

# IQR bounds
fig_comp.add_vline(x=comp_q1 - 1.5 * comp_iqr, line_dash="dash", line_color="#7209B7",
                   annotation_text="IQR lower")
fig_comp.add_vline(x=comp_q3 + 1.5 * comp_iqr, line_dash="dash", line_color="#7209B7",
                   annotation_text="IQR upper")

fig_comp.update_layout(xaxis_title=FEATURE_LABELS.get(feature_col, feature_col))
apply_common_layout(fig_comp, title="Z-Score vs IQR Bounds", height=400)
st.plotly_chart(fig_comp, use_container_width=True)

insight_box(
    "The IQR method tends to be more conservative for symmetric distributions but can disagree "
    "with Z-score for skewed data (like wind speed, which has a long right tail). "
    "When both methods flag the same point, you can be fairly confident it is truly unusual. "
    "When they disagree, it is worth investigating why."
)

st.divider()

# ---------------------------------------------------------------------------
# 5. Code Example
# ---------------------------------------------------------------------------
code_example("""
import numpy as np

data = city_df['temperature_c'].values

# Z-Score Method
mu, sigma = data.mean(), data.std()
z_scores = (data - mu) / sigma
z_anomalies = np.abs(z_scores) > 3

# IQR Method
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
iqr_anomalies = (data < lower) | (data > upper)

# Contextual (per-month)
for month in range(1, 13):
    mask = city_df['month'] == month
    month_data = city_df.loc[mask, 'temperature_c']
    mu_m, sigma_m = month_data.mean(), month_data.std()
    z_m = (month_data - mu_m) / sigma_m
    city_df.loc[mask, 'is_anomaly'] = np.abs(z_m) > 3
""")

st.divider()

# ---------------------------------------------------------------------------
# 6. Quiz
# ---------------------------------------------------------------------------
quiz(
    "A temperature of 35 deg C has a Z-score of 2.5 in the global distribution but "
    "a Z-score of 0.5 for July. This is an example of:",
    [
        "A global anomaly that is contextually normal",
        "A contextual anomaly that is globally normal",
        "An anomaly by both methods",
        "Not an anomaly by either method",
    ],
    correct_idx=0,
    explanation="35 deg C looks unusual compared to the overall distribution (z=2.5), but "
                "it is perfectly normal for July (z=0.5). This is exactly why contextual "
                "detection matters -- without it, you would flag every hot summer day.",
    key="ch57_quiz1",
)

quiz(
    "Which method is more robust to outliers in the data?",
    [
        "Z-Score -- because it uses the mean",
        "IQR -- because quartiles are resistant to extreme values",
        "Both are equally robust",
        "Neither is robust to outliers",
    ],
    correct_idx=1,
    explanation="The IQR method uses quartiles, which are robust statistics -- they do not "
                "budge when extreme values change. The Z-score uses mean and std, both of which "
                "are pulled by outliers. It is one of those cases where the simpler method has "
                "a real practical advantage.",
    key="ch57_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 7. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Z-score flags points far from the mean in standard deviation units -- intuitive but sensitive to the very outliers it is trying to detect.",
    "IQR method uses quartiles and is robust to outliers, making it a better default when you do not trust your data to be clean.",
    "Contextual anomaly detection accounts for expected variation (by season, time of day). Without context, you cannot distinguish 'unusual for January' from 'just summer.'",
    "Heat waves, cold snaps, and storms appear as anomalies in temperature, pressure, or wind data -- the interesting stories in weather are often the anomalies.",
    "Combining multiple methods increases confidence in detected anomalies. If two different approaches agree something is weird, it probably is.",
])

navigation(
    prev_label="Ch 56: Probabilistic Programming",
    prev_page="56_Probabilistic_Programming.py",
    next_label="Ch 58: Isolation Forest",
    next_page="58_Isolation_Forest.py",
)
