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
st.set_page_config(page_title="Ch 58: Isolation Forest", layout="wide")
df = load_data()
fdf = sidebar_filters(df)

chapter_header(58, "Isolation Forest", part="XIV")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "Isolation Forest Algorithm",
    "Here is a genuinely clever idea: anomalies are, by definition, few and different. "
    "So if you try to isolate a data point by randomly splitting the feature space, anomalies "
    "should require <em>fewer splits</em> to end up alone. They are the loners at the party -- "
    "easy to separate from the crowd.<br><br>"
    "Isolation Forest exploits this by building random binary trees. Each tree randomly "
    "selects a feature and a split value. Anomalous points end up with shorter path lengths "
    "(they are isolated quickly), while normal points are buried deeper in the tree.<br><br>"
    "<b>Key advantages</b>:<br>"
    "- Works in high dimensions (multivariate anomalies) without breaking a sweat<br>"
    "- Makes no distributional assumptions -- it does not care if your data is normal<br>"
    "- Linear time complexity O(n), which is impressively fast<br>"
    "- Handles mixed anomaly types without being told what to look for",
)

formula_box(
    "Anomaly Score",
    r"s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}",
    "h(x) is the path length for point x; c(n) is the average path length in a "
    "binary search tree of n points. Score close to 1 = definitely anomalous; "
    "close to 0.5 = boringly normal.",
)

st.markdown("""
### Why Isolation Forest for Weather?

The statistical methods from the last chapter (Z-score, IQR) check one variable at a time.
But some anomalies are only visible when you look at **combinations** of features. A temperature
of 30 C is fine. A humidity of 20% is fine. But 30 C *and* 20% humidity in Houston? That is
desert weather in a subtropical city -- genuinely unusual.

Consider these multivariate anomalies:

- **High wind + low pressure**: a storm system rolling in
- **High temperature + low humidity**: desert-like conditions (unusual for coastal cities)
- **Low temperature + high humidity**: fog or freezing rain conditions

None of these are extreme on any single axis. You need to look at the combination.
Isolation Forest does this naturally, because it splits on all features simultaneously.
""")

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Isolation Forest Anomaly Detection
# ---------------------------------------------------------------------------
st.subheader("Interactive: Multivariate Anomaly Detection")

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
    "The anomaly score measures how easy it is to isolate a point -- lower (more negative) "
    "scores mean the point was isolated faster, which is the algorithm's way of saying 'this "
    "one is not like the others.' The contamination parameter sets the threshold on this score, "
    "which is where your domain knowledge comes in."
)

st.divider()

# ---------------------------------------------------------------------------
# 4. Contamination Sensitivity
# ---------------------------------------------------------------------------
st.subheader("Contamination Sensitivity Analysis")

st.markdown(
    "The contamination parameter is the single most important knob you get to turn. "
    "It tells the algorithm what fraction of your data you expect to be anomalous. "
    "Set it too high and you get false positives. Set it too low and you miss real anomalies. "
    "There is no free lunch here -- you need domain knowledge."
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
    "Contamination is not something the algorithm learns from data -- you must set it based on "
    "domain knowledge. This is both a weakness (you need to know something in advance) and a "
    "strength (you are encoding real-world understanding). In weather data, 1-5% is usually "
    "reasonable. If your anomaly detector flags 20% of your data, either something catastrophic "
    "happened to the climate or you need to recalibrate."
)

st.divider()

# ---------------------------------------------------------------------------
# 5. Compare to Z-Score Method
# ---------------------------------------------------------------------------
st.subheader("Comparison: Isolation Forest vs Z-Score")

st.markdown(
    "Here is where things get interesting. We compare the multivariate Isolation Forest "
    "to a simple univariate Z-score applied to each feature separately. The points "
    "they disagree about reveal what the multivariate approach buys you."
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
    "The purple diamonds (IF Only) are the multivariate anomalies that Z-score misses -- "
    "points where no single feature is extreme but the *combination* is unusual. "
    "The orange squares (Z-Score Only) are points that look extreme on one axis but are "
    "actually normal in the broader context of all four features. This is the fundamental "
    "tradeoff: univariate methods are simple but miss interactions; multivariate methods "
    "catch them but require more careful tuning."
)

st.divider()

# ---------------------------------------------------------------------------
# 6. Feature Importance for Anomalies
# ---------------------------------------------------------------------------
st.subheader("What Makes These Points Anomalous?")

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
    "What does the contamination parameter control in Isolation Forest?",
    [
        "The number of trees in the ensemble",
        "The expected proportion of anomalies in the dataset",
        "The maximum depth of each tree",
        "The learning rate",
    ],
    correct_idx=1,
    explanation="Contamination sets the expected fraction of anomalies. It determines "
                "the threshold on the anomaly score for classifying points. You need domain "
                "knowledge to set this well -- the algorithm cannot figure it out for you.",
    key="ch58_quiz1",
)

quiz(
    "Why can Isolation Forest detect anomalies that Z-score misses?",
    [
        "It uses a higher threshold",
        "It has more parameters to tune",
        "It considers feature combinations (multivariate), not just individual features",
        "It uses a non-parametric distribution",
    ],
    correct_idx=2,
    explanation="Isolation Forest operates on all features simultaneously, catching points "
                "where the combination is unusual even if no single feature is extreme. "
                "This is the whole point of multivariate anomaly detection.",
    key="ch58_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 9. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Isolation Forest detects anomalies by measuring how easy it is to isolate a point with random splits -- a beautifully simple idea that works surprisingly well.",
    "Anomalous points have shorter average path lengths (they are isolated more quickly). Normal points blend into the crowd.",
    "The contamination parameter must be set based on domain knowledge. The algorithm tells you which points are most unusual; you decide how many to call anomalies.",
    "Unlike Z-score, Isolation Forest naturally handles multivariate anomalies -- combinations of features that are unusual even when individual features are not.",
    "Comparing methods helps identify which anomalies are most reliably unusual. When Isolation Forest and Z-score agree, you can be especially confident.",
])

navigation(
    prev_label="Ch 57: Statistical Anomaly Detection",
    prev_page="57_Statistical_Anomaly_Detection.py",
    next_label="Ch 59: Autoencoder Anomaly Detection",
    next_page="59_Autoencoder_Anomaly_Detection.py",
)
