"""Chapter 58: Isolation Forest — Contamination parameter, multivariate anomalies."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
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

chapter_header(58, "Isolation Forest", part="XIV")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
st.markdown(
    "In the last chapter, we detected anomalies by looking at one weather feature at a "
    "time: is this temperature reading unusually high? Is this wind speed unusually fast? "
    "That works great when the anomaly is extreme on a single axis. But here is a scenario "
    "where it completely fails."
)
st.markdown(
    "**The problem**: A reading comes in from Houston -- temperature 30 degrees C, "
    "relative humidity 20%. Run Z-score on temperature: perfectly normal for a Texas "
    "summer. Run Z-score on humidity: 20% is low but not unheard of. Neither feature "
    "triggers an alarm individually. But *both together*? That is desert weather in a "
    "subtropical coastal city. Houston at 30 degrees C should have humidity around "
    "65-80%, not 20%. The *combination* is the anomaly, not either feature alone."
)
st.markdown(
    "This is a **multivariate anomaly** -- unusual in the interaction between features, "
    "not in any single feature. To catch these, we need an algorithm that looks at all "
    "features simultaneously. Enter Isolation Forest."
)

concept_box(
    "How Isolation Forest Works (The Loner-at-the-Party Intuition)",
    "Here is a genuinely clever idea. Think about trying to isolate a specific person "
    "in a crowd by asking random yes/no questions. 'Are you taller than 180 cm? Are you "
    "wearing a red shirt? Are you standing in the left half of the room?'<br><br>"
    "If someone is in the middle of a dense group, you need many questions to single "
    "them out. But if someone is standing alone in the corner -- the anomaly -- you can "
    "isolate them with very few questions. 'Are you in the northeast corner?' Done.<br><br>"
    "Isolation Forest does exactly this with data. It builds random binary trees that "
    "split the 4-dimensional weather space (temperature, humidity, wind, pressure) with "
    "random cuts. An anomalous reading -- say, that Houston 30 degrees C / 20% humidity "
    "combo -- ends up alone after just 2-3 splits. A normal reading (Houston, 30 degrees C, "
    "72% humidity) needs 8-10 splits because it is surrounded by similar points.<br><br>"
    "<b>The anomaly score is just the average path length across many random trees.</b> "
    "Short paths = easy to isolate = anomalous. Long paths = hard to isolate = normal.<br><br>"
    "<b>Key advantages</b>:<br>"
    "- Works on all 4 weather features simultaneously (catches multivariate anomalies)<br>"
    "- Makes no assumptions about how the data is distributed<br>"
    "- Runs in O(n) time -- fast enough for our 105,000 rows<br>"
    "- Does not need you to specify what 'normal' looks like in advance",
)

formula_box(
    "Anomaly Score",
    r"s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}",
    "h(x) is the path length for point x (how many splits to isolate it); "
    "c(n) is the average path length in a random binary tree of n points (the baseline). "
    "Score close to 1 = definitely anomalous (isolated quickly); close to 0.5 = boringly "
    "normal (takes the expected number of splits). For that Houston desert-weather reading, "
    "we would expect a score near 0.7-0.9.",
)

st.markdown("""
### Why This Matters for Our Weather Data

The Z-score and IQR methods from the last chapter check one variable at a time.
Here are multivariate anomalies they would miss entirely:

- **30 degrees C + 20% humidity in Houston**: each value is individually normal, but the
  combination is desert weather in a subtropical city
- **High wind (40 km/h) + dropping pressure (990 hPa)**: a storm system rolling in --
  neither value alone is extreme, but together they signal severe weather
- **0 degrees C + 95% humidity + 0 km/h wind**: fog or freezing rain conditions, where
  the combination matters more than any single reading

Isolation Forest catches all of these because it splits on all features simultaneously.
When it tries to isolate that Houston desert-weather reading, it lands in an empty region
of the 4D feature space after just a few splits -- because no other Houston readings live
in the low-humidity corner of that space.
""")

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Isolation Forest Anomaly Detection
# ---------------------------------------------------------------------------
st.subheader("Interactive: Multivariate Anomaly Detection")

st.markdown(
    "Now you get to run Isolation Forest on real weather data. Here is what each control does:"
)
st.markdown(
    "- **City**: Different cities have different 'normal' weather, so anomalies differ. "
    "LA has very narrow weather patterns (anomalies are subtle); Dallas has wide variation "
    "(anomalies need to be truly extreme).\n"
    "- **Contamination**: The single most important parameter. It tells the algorithm what "
    "fraction of your data you expect to be anomalous. Set it to 0.02 and exactly 2% of "
    "readings get flagged. Too high = false alarms; too low = missed anomalies.\n"
    "- **Number of trees**: More trees = more stable results, but slower. 100 is usually "
    "sufficient; going to 500 gives marginal improvement.\n"
    "- **Scatter plot axes**: Choose which two features to visualize. The algorithm always "
    "uses all 4 features internally -- but we can only see 2 at a time on screen. Try "
    "different pairs to understand why specific points were flagged."
)

col_ctrl, col_viz = st.columns([1, 3])

with col_ctrl:
    if_city = st.selectbox("City", CITY_LIST, key="if_city")
    contamination = st.slider(
        "Contamination (expected anomaly fraction)",
        0.001, 0.10, 0.02, 0.001,
        format="%.3f",
        key="if_contam",
        help="Fraction of data expected to be anomalous",
    )
    n_estimators = st.slider("Number of trees", 50, 500, 100, 50, key="if_trees")

    st.markdown("---")
    st.markdown("**Scatter plot axes**")
    x_feat = st.selectbox(
        "X-axis", FEATURE_COLS,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
        index=2,  # wind_speed_kmh
        key="if_xfeat",
    )
    y_feat = st.selectbox(
        "Y-axis", FEATURE_COLS,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
        index=3,  # surface_pressure_hpa
        key="if_yfeat",
    )

city_data = fdf[fdf["city"] == if_city][FEATURE_COLS + ["datetime", "month", "season"]].dropna().copy()
if len(city_data) < 100:
    st.warning("Not enough data. Adjust filters.")
    st.stop()

# Subsample for performance if needed
max_points = 10000
if len(city_data) > max_points:
    rng = np.random.RandomState(42)
    city_data = city_data.sample(n=max_points, random_state=42)

# Fit Isolation Forest
@st.cache_data(show_spinner="Training Isolation Forest...")
def fit_isolation_forest(data_values, contamination, n_estimators, seed=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_values)
    iso = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=seed,
    )
    preds = iso.fit_predict(X_scaled)
    scores = iso.decision_function(X_scaled)
    return preds, scores

preds, scores = fit_isolation_forest(
    city_data[FEATURE_COLS].values, contamination, n_estimators
)

city_data["iso_anomaly"] = preds == -1
city_data["iso_score"] = scores

n_if_anomalies = city_data["iso_anomaly"].sum()

with col_viz:
    # 2D scatter
    fig = go.Figure()

    normal_mask = ~city_data["iso_anomaly"]
    anom_mask = city_data["iso_anomaly"]

    fig.add_trace(go.Scatter(
        x=city_data.loc[normal_mask, x_feat],
        y=city_data.loc[normal_mask, y_feat],
        mode="markers",
        marker=dict(color=CITY_COLORS.get(if_city, "#2A9D8F"), size=3, opacity=0.3),
        name="Normal",
    ))

    fig.add_trace(go.Scatter(
        x=city_data.loc[anom_mask, x_feat],
        y=city_data.loc[anom_mask, y_feat],
        mode="markers",
        marker=dict(color="#E63946", size=8, symbol="x", line=dict(width=1)),
        name=f"Anomaly ({n_if_anomalies})",
    ))

    fig.update_layout(
        xaxis_title=FEATURE_LABELS.get(x_feat, x_feat),
        yaxis_title=FEATURE_LABELS.get(y_feat, y_feat),
    )
    apply_common_layout(
        fig,
        title=f"Isolation Forest: {FEATURE_LABELS[x_feat]} vs {FEATURE_LABELS[y_feat]} ({if_city})",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Points", f"{len(city_data):,}")
m2.metric("Anomalies", f"{n_if_anomalies:,}")
m3.metric("Anomaly Rate", f"{n_if_anomalies / len(city_data) * 100:.2f}%")
m4.metric("Target Rate", f"{contamination * 100:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# 3. Anomaly Score Distribution
# ---------------------------------------------------------------------------
st.subheader("Anomaly Score Distribution")

st.markdown(
    "Every data point gets an anomaly score from the Isolation Forest. The score is based "
    "on how quickly the point was isolated: more negative = faster isolation = more anomalous. "
    "The decision boundary (the vertical dashed line at 0) separates normal from anomalous. "
    "Points to the left of the line were easy to isolate -- the algorithm is saying 'this "
    "weather reading does not look like the others.'"
)

fig_score = go.Figure()
fig_score.add_trace(go.Histogram(
    x=city_data.loc[normal_mask, "iso_score"], nbinsx=80,
    name="Normal", marker_color="#2A9D8F", opacity=0.7,
))
fig_score.add_trace(go.Histogram(
    x=city_data.loc[anom_mask, "iso_score"], nbinsx=30,
    name="Anomaly", marker_color="#E63946", opacity=0.7,
))
fig_score.add_vline(x=0, line_dash="dash", line_color="black",
                    annotation_text="Decision Boundary")
fig_score.update_layout(
    xaxis_title="Anomaly Score (negative = more anomalous)",
    yaxis_title="Count", barmode="overlay",
)
apply_common_layout(fig_score, title="Isolation Forest Anomaly Score Distribution", height=400)
st.plotly_chart(fig_score, use_container_width=True)

insight_box(
    "Look at the overlap region near the decision boundary. Points right at the edge "
    "are borderline -- the algorithm is not very confident about them. Points far to "
    "the left (scores of -0.2 or lower) were isolated extremely quickly, meaning their "
    "combination of temperature, humidity, wind, and pressure is genuinely unusual for "
    "this city. The contamination parameter determines exactly where on this score "
    "distribution the boundary gets placed -- that is where your domain knowledge enters."
)

st.divider()

# ---------------------------------------------------------------------------
# 4. Contamination Sensitivity
# ---------------------------------------------------------------------------
st.subheader("Contamination Sensitivity Analysis")

st.markdown(
    "The contamination parameter is the single most important knob you get to turn, and "
    "the algorithm cannot figure it out for you. Here is the practical question: what "
    "fraction of weather readings in Houston do you expect to be genuinely anomalous? "
    "If you answer '0.5%,' that means about 88 out of 17,500 hourly readings. If you "
    "answer '5%,' that is 875 readings -- nearly one per day. The difference is enormous, "
    "and there is no mathematical way to determine the right answer. You need to know "
    "something about weather to set this well."
)

contam_values = [0.005, 0.01, 0.02, 0.05, 0.10]
contam_results = []
for c in contam_values:
    p, s = fit_isolation_forest(city_data[FEATURE_COLS].values, c, 100)
    n_anom = (p == -1).sum()
    contam_results.append({
        "Contamination": f"{c:.3f}",
        "Anomalies Detected": n_anom,
        "Percentage": f"{n_anom / len(city_data) * 100:.2f}%",
    })

st.dataframe(pd.DataFrame(contam_results), use_container_width=True, hide_index=True)

fig_contam = go.Figure()
fig_contam.add_trace(go.Bar(
    x=[r["Contamination"] for r in contam_results],
    y=[r["Anomalies Detected"] for r in contam_results],
    marker_color=["#2A9D8F", "#F4A261", "#E63946", "#7209B7", "#FB8500"],
))
fig_contam.update_layout(xaxis_title="Contamination", yaxis_title="Anomalies Detected")
apply_common_layout(fig_contam, title="Anomaly Count vs Contamination", height=350)
st.plotly_chart(fig_contam, use_container_width=True)

warning_box(
    "A contamination of 0.02 (2%) for weather data means you expect about 1 in 50 "
    "hourly readings to be genuinely anomalous. That translates to roughly one anomalous "
    "reading every two days -- which feels about right for interesting weather events like "
    "unusual temperature-humidity combos, storm signatures, or sensor glitches. At 0.10 "
    "(10%), you are flagging one in every ten readings, which would mean 2-3 per day -- "
    "at that point, 'anomalous' has become 'somewhat unusual,' and you have diluted the "
    "signal. If your anomaly detector flags 20% of your data, either something catastrophic "
    "happened to the climate or you need to recalibrate."
)

st.divider()

# ---------------------------------------------------------------------------
# 5. Compare to Z-Score Method
# ---------------------------------------------------------------------------
st.subheader("Comparison: Isolation Forest vs Z-Score")

st.markdown(
    "Here is where things get interesting. We run both the multivariate Isolation Forest "
    "(which looks at all 4 features simultaneously) and a simple univariate Z-score "
    "(applied to each feature separately, flagging any reading where |z| > 3 on *any* "
    "feature). The points they disagree about reveal exactly what the multivariate approach buys you."
)

# Z-score anomalies (|z| > 3 on any feature)
z_anomaly_mask = np.zeros(len(city_data), dtype=bool)
for feat in FEATURE_COLS:
    vals = city_data[feat].values
    mu, sigma = vals.mean(), vals.std()
    z = np.abs((vals - mu) / (sigma if sigma > 0 else 1))
    z_anomaly_mask |= z > 3

city_data["z_anomaly"] = z_anomaly_mask
n_z = z_anomaly_mask.sum()

# Overlap analysis
both = city_data["iso_anomaly"] & city_data["z_anomaly"]
iso_only = city_data["iso_anomaly"] & ~city_data["z_anomaly"]
z_only = ~city_data["iso_anomaly"] & city_data["z_anomaly"]

overlap_df = pd.DataFrame({
    "Category": [
        "Isolation Forest Only",
        "Z-Score Only",
        "Both Methods",
        "Total IF Anomalies",
        "Total Z-Score Anomalies",
    ],
    "Count": [iso_only.sum(), z_only.sum(), both.sum(), n_if_anomalies, n_z],
})
st.dataframe(overlap_df, use_container_width=True, hide_index=True)

# Plot showing agreement / disagreement
fig_comp = go.Figure()

# Normal for both
neither = ~city_data["iso_anomaly"] & ~city_data["z_anomaly"]
fig_comp.add_trace(go.Scatter(
    x=city_data.loc[neither, x_feat],
    y=city_data.loc[neither, y_feat],
    mode="markers", marker=dict(color="#ccc", size=2, opacity=0.2),
    name="Normal (both)",
))

if both.sum() > 0:
    fig_comp.add_trace(go.Scatter(
        x=city_data.loc[both, x_feat],
        y=city_data.loc[both, y_feat],
        mode="markers", marker=dict(color="#E63946", size=8, symbol="circle"),
        name=f"Both ({both.sum()})",
    ))

if iso_only.sum() > 0:
    fig_comp.add_trace(go.Scatter(
        x=city_data.loc[iso_only, x_feat],
        y=city_data.loc[iso_only, y_feat],
        mode="markers", marker=dict(color="#7209B7", size=7, symbol="diamond"),
        name=f"IF Only ({iso_only.sum()})",
    ))

if z_only.sum() > 0:
    fig_comp.add_trace(go.Scatter(
        x=city_data.loc[z_only, x_feat],
        y=city_data.loc[z_only, y_feat],
        mode="markers", marker=dict(color="#FB8500", size=7, symbol="square"),
        name=f"Z-Score Only ({z_only.sum()})",
    ))

fig_comp.update_layout(
    xaxis_title=FEATURE_LABELS.get(x_feat, x_feat),
    yaxis_title=FEATURE_LABELS.get(y_feat, y_feat),
)
apply_common_layout(fig_comp, title="Method Comparison: Isolation Forest vs Z-Score", height=500)
st.plotly_chart(fig_comp, use_container_width=True)

insight_box(
    "The purple diamonds (IF Only) are the whole point of this chapter. These are "
    "readings where no single feature exceeds the 3-sigma threshold, but the *combination* "
    "of features is unusual. That Houston reading of 30 degrees C and 20% humidity would "
    "show up here -- normal temperature, normal-ish humidity, but the pair is desert "
    "weather in a coastal city. The orange squares (Z-Score Only) are the opposite: "
    "points where one feature is individually extreme (say, a 45-degree C spike) but "
    "the full 4D picture is actually consistent with what the Isolation Forest has seen "
    "before. Switch between different axis pairs to see which feature combinations the "
    "Isolation Forest is keying on."
)

st.divider()

# ---------------------------------------------------------------------------
# 6. Feature Importance for Anomalies
# ---------------------------------------------------------------------------
st.subheader("What Makes These Points Anomalous?")

st.markdown(
    "When Isolation Forest flags a reading, a natural question is: *which* features made "
    "it unusual? We can answer this by comparing the average feature values of flagged "
    "anomalies against the normal readings. If anomalies have much higher wind speed "
    "on average, wind is probably driving those detections. If they have unusual humidity, "
    "that is the key feature."
)

if n_if_anomalies > 0:
    anom_data = city_data[city_data["iso_anomaly"]]
    norm_data = city_data[~city_data["iso_anomaly"]]

    comparison_rows = []
    for feat in FEATURE_COLS:
        comparison_rows.append({
            "Feature": FEATURE_LABELS.get(feat, feat),
            "Normal Mean": f"{norm_data[feat].mean():.2f}",
            "Normal Std": f"{norm_data[feat].std():.2f}",
            "Anomaly Mean": f"{anom_data[feat].mean():.2f}",
            "Anomaly Std": f"{anom_data[feat].std():.2f}",
            "Difference (in std)": f"{abs(anom_data[feat].mean() - norm_data[feat].mean()) / norm_data[feat].std():.2f}",
        })

    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    st.markdown(
        "The 'Difference (in std)' column tells you how far apart the anomaly group is "
        "from normal, measured in standard deviations. A value of 1.0 or higher means "
        "anomalies are systematically different on that feature. A value near 0 means "
        "that feature is not driving the detections."
    )

    # Box plots comparing normal vs anomaly
    fig_box = make_subplots(rows=1, cols=len(FEATURE_COLS),
                            subplot_titles=[FEATURE_LABELS[f] for f in FEATURE_COLS])

    for i, feat in enumerate(FEATURE_COLS):
        fig_box.add_trace(
            go.Box(y=norm_data[feat], name="Normal", marker_color="#2A9D8F",
                   showlegend=(i == 0)),
            row=1, col=i + 1,
        )
        fig_box.add_trace(
            go.Box(y=anom_data[feat], name="Anomaly", marker_color="#E63946",
                   showlegend=(i == 0)),
            row=1, col=i + 1,
        )

    fig_box.update_layout(template="plotly_white", height=400, margin=dict(t=40, b=40))
    st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 7. Code Example
# ---------------------------------------------------------------------------
code_example("""
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

features = ['temperature_c', 'relative_humidity_pct',
            'wind_speed_kmh', 'surface_pressure_hpa']
X = city_df[features].dropna()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Isolation Forest
iso = IsolationForest(
    contamination=0.02,  # expect 2% anomalies
    n_estimators=100,
    random_state=42
)
predictions = iso.fit_predict(X_scaled)  # 1 = normal, -1 = anomaly
scores = iso.decision_function(X_scaled)  # lower = more anomalous

# Flag anomalies
city_df['is_anomaly'] = predictions == -1
city_df['anomaly_score'] = scores

print(f"Anomalies: {(predictions == -1).sum()} / {len(X)}")
""")

st.divider()

# ---------------------------------------------------------------------------
# 8. Quiz
# ---------------------------------------------------------------------------
quiz(
    "Houston logs a reading of temperature = 30 degrees C, humidity = 18%, wind = 12 km/h, "
    "pressure = 1013 hPa. The Z-score for each feature is below 2. Isolation Forest flags "
    "it as anomalous. What is the most likely explanation?",
    [
        "Isolation Forest is broken and should be recalibrated",
        "The contamination parameter is set too high",
        "The combination of 30 degrees C and 18% humidity is unusual for Houston, even though "
        "each value individually is within normal range",
        "Z-score is always more accurate than Isolation Forest",
    ],
    correct_idx=2,
    explanation="This is the textbook case for multivariate anomaly detection. Houston is a "
                "humid subtropical city where 30 degrees C typically comes with 65-80% humidity. "
                "A reading of 18% humidity at that temperature means unusually dry air -- desert-like "
                "conditions that are rare in Houston. Z-score checks each feature against its own "
                "distribution and finds nothing extreme. Isolation Forest checks the 4D combination "
                "and correctly identifies that this region of feature space (warm + bone dry) is "
                "nearly empty for Houston data. This is exactly what multivariate anomaly detection "
                "is designed to catch.",
    key="ch58_quiz1",
)

quiz(
    "What does the contamination parameter control in Isolation Forest?",
    [
        "The number of trees in the ensemble",
        "The expected proportion of anomalies in the dataset",
        "The maximum depth of each tree",
        "The learning rate for tree construction",
    ],
    correct_idx=1,
    explanation="Contamination tells the algorithm what fraction of readings you expect to be "
                "anomalous. Set it to 0.02 and the algorithm places its decision boundary so that "
                "exactly 2% of points fall on the anomalous side. For our weather data, 0.02 means "
                "about 1 anomalous reading every two days -- perhaps a storm signature, an unusual "
                "temperature-humidity combination, or a sensor glitch. The algorithm ranks all points "
                "by how easy they were to isolate; contamination just determines where to draw the line "
                "between 'unusual' and 'anomalous.' You need domain knowledge to set this well.",
    key="ch58_quiz2",
)

quiz(
    "You run Isolation Forest on Dallas weather with contamination=0.02 (2%). The results show "
    "that 85% of flagged anomalies occur during December-February. Is this a problem?",
    [
        "Yes -- the model is biased toward winter data and should be retrained",
        "Yes -- contamination should be increased to spread anomalies evenly across seasons",
        "Not necessarily -- Dallas winters are when extreme weather events (ice storms, cold "
        "snaps, unusual pressure patterns) are most likely, so this concentration makes physical sense",
        "Not necessarily -- 2% is too low and we need at least 10%",
    ],
    correct_idx=2,
    explanation="Anomalies should NOT be evenly distributed across seasons unless weather extremes "
                "are equally likely year-round, which they are not. Dallas winters bring ice storms, "
                "cold fronts pushing Arctic air into Texas, and unusual pressure patterns -- exactly "
                "the kind of multivariate anomalies Isolation Forest is designed to detect. The "
                "concentration in winter is the model correctly identifying when unusual weather "
                "actually happens. If anomalies were evenly spread, that would be more suspicious, "
                "suggesting the model is picking up noise rather than real events.",
    key="ch58_quiz3",
)

st.divider()

# ---------------------------------------------------------------------------
# 9. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Isolation Forest detects anomalies by measuring how easy it is to isolate a point with "
    "random splits. That Houston reading of 30 degrees C / 20% humidity gets isolated in 2-3 "
    "splits because no other Houston readings live in that warm-and-dry corner of feature space.",
    "The key advantage over Z-score: Isolation Forest catches multivariate anomalies where "
    "no single feature is extreme but the combination is unusual. Temperature is fine, humidity "
    "is fine, but temperature AND humidity together in this city? That does not happen.",
    "The contamination parameter is your domain knowledge knob. For weather data, 1-3% is "
    "usually reasonable -- that is roughly 1 anomalous event per day or every few days, which "
    "aligns with how often genuinely unusual weather occurs.",
    "When Isolation Forest and Z-score agree on an anomaly, you can be especially confident. "
    "When they disagree, the disagreements teach you something: IF-only anomalies are "
    "multivariate patterns Z-score misses; Z-only anomalies are single-feature extremes that "
    "are normal in the broader context.",
    "Always inspect your anomalies after detection. If the flagged readings do not look unusual "
    "to a domain expert (or if they all cluster suspiciously), recalibrate your contamination "
    "or check whether the model is picking up seasonal variation instead of true anomalies.",
])

navigation(
    prev_label="Ch 57: Statistical Anomaly Detection",
    prev_page="57_Statistical_Anomaly_Detection.py",
    next_label="Ch 59: Autoencoder Anomaly Detection",
    next_page="59_Autoencoder_Anomaly_Detection.py",
)
