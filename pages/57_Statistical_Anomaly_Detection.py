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
df = load_data()
fdf = sidebar_filters(df)

chapter_header(57, "Statistical Anomaly Detection", part="XIV")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
st.markdown(
    "Let me set up a specific problem. You are monitoring hourly weather data "
    "from 6 US cities -- Dallas, Houston, Austin, San Antonio, NYC, and Los Angeles. "
    "That is about 105,000 rows of temperature, humidity, wind speed, and pressure "
    "readings, one per hour, for two years."
)
st.markdown(
    "A reading comes in: **35 degrees C in New York City on January 15th at 2 PM.** "
    "Is that weird? Your gut says yes -- obviously yes, that is summer weather in the "
    "dead of winter. But 'my gut says yes' is not a statistical method. We need a "
    "rigorous, automated way to answer the question 'is this data point weird?' for "
    "every one of those 105,000 readings, across all four features, without a human "
    "staring at each one."
)
st.markdown(
    "That is anomaly detection. And the tricky part is defining 'weird' precisely "
    "enough for a computer to compute it."
)

concept_box(
    "Why We Care About Weather Anomalies",
    "In our dataset, anomalies are the interesting stuff -- the events a meteorologist "
    "would actually want to investigate:<br><br>"
    "- <b>Heat waves</b>: Dallas hitting 42 degrees C in July (even for Texas, that is extreme)<br>"
    "- <b>Cold snaps</b>: Dallas dropping to -15 degrees C in February 2021, which residents "
    "would enthusiastically confirm was indeed unusual<br>"
    "- <b>Storm signatures</b>: a sudden 20 hPa pressure drop in Houston over 6 hours, "
    "paired with 80+ km/h winds<br>"
    "- <b>Sensor errors</b>: a humidity reading of 0% in Houston (which is physically "
    "implausible for a subtropical coastal city)<br><br>"
    "The tricky part: 35 degrees C in Dallas in July is a normal Tuesday. 35 degrees C "
    "in NYC in January is front-page news. The same number can be normal or anomalous "
    "depending on <b>context</b> -- and we need our methods to handle that.",
)

st.markdown("### Two Classical Methods")
st.markdown(
    "There are two simple, well-understood approaches that have been used for decades. "
    "They differ in a fundamental way: one uses the mean and standard deviation (which "
    "are themselves affected by outliers), and the other uses quartiles (which are not). "
    "Both have the same goal -- draw a boundary around 'normal' and flag anything "
    "outside it."
)

col1, col2 = st.columns(2)

with col1:
    concept_box(
        "Z-Score Method",
        "Here is the idea in weather terms. Take all 17,500 hourly temperature readings "
        "for Dallas. Compute the mean (say, 19.2 degrees C) and the standard deviation (say, "
        "9.8 degrees C). Now for any single reading, ask: how many standard deviations is this "
        "from the mean?<br><br>"
        "A reading of 48.6 degrees C would be (48.6 - 19.2) / 9.8 = 3.0 standard deviations "
        "above the mean. At a threshold of 3, that is flagged as an anomaly. A reading of "
        "25 degrees C would be (25 - 19.2) / 9.8 = 0.6 -- perfectly normal.<br><br>"
        "<b>The catch</b>: the mean and standard deviation are <em>themselves</em> influenced "
        "by outliers. If Dallas had one freak reading of 60 degrees C, that would pull the mean "
        "up and inflate the standard deviation, making future anomalies <em>harder</em> to "
        "detect. You are asking the suspects to serve on the jury.<br><br>"
        "<b>Pros</b>: Simple, intuitive, works well when data is roughly bell-shaped.<br>"
        "<b>Cons</b>: Sensitive to the very outliers it is trying to detect. Assumes normality.",
    )
    formula_box(
        "Z-Score",
        r"z = \frac{x - \mu}{\sigma}",
        "A z-score of 3 means the point is 3 standard deviations from the mean -- "
        "expected to happen about 0.3% of the time under a normal distribution. "
        "For Dallas temperature, that means roughly 50 out of 17,500 readings.",
    )

with col2:
    concept_box(
        "IQR Method",
        "Same dataset, different approach. Sort all 17,500 Dallas temperature readings. "
        "Find the 25th percentile (Q1, say 11.4 degrees C) and the 75th percentile (Q3, "
        "say 27.1 degrees C). The interquartile range (IQR) is Q3 - Q1 = 15.7 degrees C. "
        "That is the middle 50% of your data.<br><br>"
        "Now define bounds: anything below Q1 - 1.5 * IQR = 11.4 - 23.6 = -12.2 degrees C "
        "or above Q3 + 1.5 * IQR = 27.1 + 23.6 = 50.7 degrees C is an anomaly.<br><br>"
        "The key advantage: <b>quartiles do not care about outliers.</b> You could replace "
        "the single hottest reading with 1000 degrees C and the IQR would not budge -- "
        "the 25th and 75th percentiles are set by the bulk of the data, not the extremes. "
        "This is why statisticians call quartiles 'robust.'<br><br>"
        "<b>Pros</b>: Robust to outliers; no normality assumption needed.<br>"
        "<b>Cons</b>: May miss subtle anomalies; the k=1.5 multiplier is essentially "
        "arbitrary (John Tukey picked it in 1977 and it stuck).",
    )
    formula_box(
        "IQR Bounds",
        r"\text{Lower} = Q_1 - k \cdot IQR, \quad \text{Upper} = Q_3 + k \cdot IQR",
        "With k=1.5, this corresponds roughly to 2.7 sigma for normal data. "
        "For Dallas temperature, the lower bound might be around -12 degrees C and "
        "the upper bound around 51 degrees C -- anything outside that range is flagged.",
    )

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Anomaly Detection on Time Series
# ---------------------------------------------------------------------------
st.subheader("Interactive: Detect Weather Anomalies")

st.markdown(
    "Now you get to try it. Pick a city and a weather feature, choose your detection "
    "method, and watch the anomalies appear in real time. Here is what each control does:"
)
st.markdown(
    "- **City**: Each city has a different climate, so the same threshold will find "
    "different anomalies. LA's temperature barely varies; Dallas is a roller coaster.\n"
    "- **Feature**: Temperature anomalies are heat waves and cold snaps. Pressure anomalies "
    "are storm systems. Wind anomalies are gales. Humidity anomalies might be sensor errors "
    "or genuinely unusual air masses.\n"
    "- **Method**: Z-Score uses mean/std; IQR uses quartiles. Try both on the same data to "
    "see where they agree and disagree.\n"
    "- **Threshold slider**: Lower threshold = more anomalies flagged (more sensitive, but "
    "more false alarms). Higher threshold = fewer anomalies (fewer false alarms, but you "
    "might miss real events).\n"
    "- **Contextual checkbox**: The most important control. Without it, the method uses "
    "one set of bounds for the entire year. With it, bounds are computed per-month -- so "
    "a 'hot' reading in January has a different threshold than a 'hot' reading in July."
)

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
        "You are now using per-month statistics, which is the right approach for weather "
        "data. Look at what happens to the bounds -- they shift with the seasons. In July, "
        "the upper bound for Dallas temperature might be 42 degrees C; in January, it might "
        "be 22 degrees C. A reading of 35 degrees C would be perfectly normal in July but "
        "would be flagged as a screaming anomaly in January. Without per-month context, "
        "both readings get compared against the same year-round bounds, which means you "
        "either miss winter anomalies or you flag every summer day. Toggle the contextual "
        "checkbox off and watch the anomalies change -- that is the difference context makes."
    )
else:
    insight_box(
        "Right now you are using global (year-round) bounds, which means a single threshold "
        "for all 12 months. This works okay for features without strong seasonality (like "
        "surface pressure), but it is a problem for temperature. A reading of 35 degrees C "
        "is normal for Dallas in July but bizarre for January -- yet with global bounds, "
        "both get the same treatment. Try enabling the 'Contextual (per-month)' checkbox "
        "to see how the bounds adapt to seasonal patterns. You should see more targeted "
        "anomaly detection, especially for temperature."
    )

st.divider()

# ---------------------------------------------------------------------------
# 3. Anomaly Distribution Analysis
# ---------------------------------------------------------------------------
st.subheader("Anomaly Distribution Analysis")

st.markdown(
    "Now that we have flagged some anomalies, the natural question is: *when* do they "
    "happen? If anomalies cluster in certain months or at certain hours, that tells us "
    "something about what is causing them. If heat-wave anomalies all appear in July-August, "
    "that is real meteorology. If anomalies are uniformly scattered, they might be sensor noise."
)

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

    st.markdown(
        "Let us put names on what we found. For temperature, anomalies above the mean "
        "are heat waves and those below are cold snaps. For pressure, low-pressure "
        "anomalies look like storm systems and high-pressure anomalies suggest unusual "
        "atmospheric blocking patterns."
    )

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
    st.markdown(
        "Here are the actual data rows flagged as anomalies. Look at the values and "
        "ask yourself: do these look genuinely unusual for this city? Check the bounds "
        "columns -- anything outside those bounds got flagged."
    )
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
    "A reasonable question: if both methods detect anomalies, do they agree? The answer "
    "is 'sometimes,' and the disagreements are instructive. Let me show you. We will run "
    "both methods on the same data with standard settings (Z-score threshold = 3, "
    "IQR multiplier k = 1.5) and see which readings they agree on and which they fight about."
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
    "Look at where the vertical lines land. For symmetric, roughly bell-shaped data "
    "(like temperature), the Z-score bounds (solid red) and IQR bounds (dashed purple) "
    "land in similar places. But for skewed data -- try wind speed, which has a long "
    "right tail because wind speeds cannot go below zero -- the bounds diverge. The IQR "
    "method is more conservative on the right tail because quartiles are not pulled by "
    "extreme values. When both methods flag the same reading (like a Dallas temperature "
    "of -15 degrees C), you can be quite confident it is genuinely unusual. When they "
    "disagree, investigate why -- it usually reveals something about the shape of your "
    "data distribution."
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
    "A temperature reading of 35 degrees C in Dallas has a Z-score of 2.5 when computed "
    "against the full-year distribution, but a Z-score of only 0.5 when computed against "
    "July data alone. Using a threshold of 3, this reading is:",
    [
        "A global anomaly that is contextually normal",
        "A contextual anomaly that is globally normal",
        "An anomaly by both methods",
        "Not an anomaly by either method",
    ],
    correct_idx=3,
    explanation="Here is the key: with a threshold of 3, a Z-score of 2.5 does NOT "
                "exceed the threshold, and 0.5 certainly does not. So this reading is "
                "not flagged by either the global or the contextual method at that threshold. "
                "It is a warm reading (2.5 sigma above the annual mean), but not extreme enough "
                "to cross the z=3 line. If you lowered the threshold to 2, then it would be "
                "flagged globally (2.5 > 2) but not contextually (0.5 < 2) -- making it a "
                "'global anomaly that is contextually normal.' This is exactly why the threshold "
                "setting matters so much.",
    key="ch57_quiz1",
)

quiz(
    "Dallas has an extreme cold snap that drops temperatures to -18 degrees C. Which method "
    "is more likely to correctly flag this as anomalous even if there are OTHER extreme "
    "values in the dataset that inflate the spread?",
    [
        "Z-Score -- because it uses the mean, which is pulled toward extremes",
        "IQR -- because quartiles are resistant to extreme values",
        "Both are equally robust",
        "Neither can handle extreme values",
    ],
    correct_idx=1,
    explanation="The IQR method uses the 25th and 75th percentiles, which are 'robust' "
                "statistics -- they do not budge when extreme values change. If the dataset "
                "has other extreme readings that inflate the standard deviation, the Z-score's "
                "denominator gets larger, making the cold snap's Z-score *smaller* and potentially "
                "causing it to be missed. Meanwhile, the IQR bounds stay anchored to the bulk "
                "of the data. Imagine you have one reading of 60 degrees C from a sensor error "
                "-- the Z-score method's sigma would increase, making -18 degrees C look less "
                "extreme. The IQR method would not even notice the sensor error when computing "
                "its bounds.",
    key="ch57_quiz2",
)

quiz(
    "You run anomaly detection on Houston humidity with global (non-contextual) bounds and "
    "find that 80% of the flagged anomalies occur in December-February. What is the most "
    "likely explanation?",
    [
        "Houston has worse sensors in winter",
        "Winter humidity in Houston is genuinely unusual relative to the year-round average, "
        "but seasonal variation (not true anomalies) is causing the flags",
        "The IQR method is broken for humidity data",
        "Houston weather is more unpredictable in winter",
    ],
    correct_idx=1,
    explanation="Houston is a humid subtropical city where summer humidity is very high. "
                "When you compute year-round bounds, they are anchored by the high-humidity "
                "summer months. Winter months, when Houston is naturally drier, fall outside "
                "those bounds -- not because anything unusual is happening, but because the "
                "global method does not account for normal seasonal variation. This is exactly "
                "the problem that contextual (per-month) detection solves: by computing "
                "separate bounds for each month, winter readings are compared to other winter "
                "readings, not to summer.",
    key="ch57_quiz3",
)

st.divider()

# ---------------------------------------------------------------------------
# 7. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Z-score measures 'how many standard deviations from the mean' -- for Dallas temperature, "
    "a Z-score of 3 might mean the reading is about 30 degrees above or below average. "
    "Simple and intuitive, but sensitive to the very outliers it is trying to detect.",
    "IQR method uses quartiles (the 25th and 75th percentiles) and is immune to extreme "
    "values. If you suspect your data has sensor errors or corrupted readings, IQR is the "
    "safer default because those bad readings cannot pull the bounds.",
    "Contextual detection is essential for seasonal data. Without it, every Houston summer "
    "afternoon gets flagged as unusually humid, and every Dallas winter night gets flagged as "
    "unusually cold -- neither of which is actually anomalous for the season.",
    "When Z-score and IQR agree on an anomaly (like that -15 degrees C Dallas reading), "
    "you can be especially confident. When they disagree, it usually means the data is "
    "skewed, and you should investigate the distribution shape.",
    "The anomaly rate should make domain sense. If your detector flags 20% of readings "
    "as anomalous, either the climate has gone haywire or your threshold is too low.",
])

navigation(
    prev_label="Ch 56: Probabilistic Programming",
    prev_page="56_Probabilistic_Programming.py",
    next_label="Ch 58: Isolation Forest",
    next_page="58_Isolation_Forest.py",
)
