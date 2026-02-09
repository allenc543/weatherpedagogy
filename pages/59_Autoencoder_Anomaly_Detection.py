"""Chapter 59: Autoencoder Anomaly Detection — Reconstruction error as anomaly score."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

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
st.set_page_config(page_title="Ch 59: Autoencoder Anomaly Detection", layout="wide")
df = load_data()
fdf = sidebar_filters(df)

chapter_header(59, "Autoencoder Anomaly Detection", part="XIV")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "Autoencoders for Anomaly Detection",
    "An <b>autoencoder</b> is a neural network with a very specific job: learn to copy its "
    "input to its output. 'Wait,' you might say, 'that sounds trivially easy.' And it would "
    "be, except we force the data through a <b>bottleneck</b> -- a narrow hidden layer that "
    "can only represent a compressed version of the input.<br><br>"
    "<b>The clever part</b>: train the autoencoder on <em>normal</em> data. It learns to "
    "compress and reconstruct normal patterns efficiently. When an anomaly comes along -- "
    "a pattern the network has never seen -- it reconstructs it poorly, producing high "
    "reconstruction error. That error becomes your anomaly score.<br><br>"
    "<b>Architecture</b>: Input --> Encoder --> Bottleneck --> Decoder --> Output<br>"
    "The bottleneck is doing the real work. It forces the network to learn a compact "
    "representation of what 'normal weather' looks like. Anything that does not fit "
    "that representation sticks out.",
)

formula_box(
    "Reconstruction Error (Anomaly Score)",
    r"\text{RE}(x) = \|x - \hat{x}\|^2 = \sum_{i=1}^{d} (x_i - \hat{x}_i)^2",
    "x is the original input, x-hat is the reconstruction. "
    "Points with RE above a threshold are flagged as anomalies. "
    "In plain English: how badly did the network fail to recreate this data point?",
)

st.markdown("""
### Why Autoencoders for Weather Anomalies?

You might ask: if Isolation Forest already handles multivariate anomalies, why do we need
another method? Three reasons:

1. **Non-linear patterns**: Unlike PCA or even Isolation Forest, autoencoders can capture
   non-linear relationships between features (e.g., the specific curve of how temperature
   and humidity interact). They learn the *manifold* of normal data.
2. **Learned normality**: The network builds an internal model of what "normal weather" looks
   like across all features simultaneously. It is essentially learning the physics of
   typical weather, compressed into a few neurons.
3. **Flexible threshold**: You control the sensitivity via the reconstruction error threshold,
   and you can examine *which features* are hardest to reconstruct for each anomaly.
""")

warning_box(
    "We use sklearn's MLPRegressor as a simple autoencoder (input = output training target). "
    "For production use, you would want PyTorch or TensorFlow with more control "
    "over architecture, regularisation, and training. But the principle is identical."
)

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Autoencoder Training & Anomaly Detection
# ---------------------------------------------------------------------------
st.subheader("Interactive: Autoencoder Anomaly Detection")

col_ctrl, col_viz = st.columns([1, 3])

with col_ctrl:
    ae_city = st.selectbox("City", CITY_LIST, key="ae_city")
    bottleneck_size = st.slider(
        "Bottleneck size (encoding dimension)", 1, 3, 2, key="ae_bottleneck",
        help="Smaller bottleneck = more compression = stricter normality model",
    )
    threshold_pctile = st.slider(
        "Anomaly threshold (percentile)", 90.0, 99.9, 97.0, 0.1,
        key="ae_threshold",
        help="Points above this percentile of reconstruction error are anomalies",
    )

    st.markdown("---")
    x_feat = st.selectbox(
        "X-axis", FEATURE_COLS,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
        index=0, key="ae_xfeat",
    )
    y_feat = st.selectbox(
        "Y-axis", FEATURE_COLS,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
        index=1, key="ae_yfeat",
    )

city_data = fdf[fdf["city"] == ae_city][FEATURE_COLS + ["datetime", "month"]].dropna().copy()
if len(city_data) < 200:
    st.warning("Not enough data. Adjust filters.")
    st.stop()

# Subsample for performance
max_points = 8000
if len(city_data) > max_points:
    city_data = city_data.sample(n=max_points, random_state=42)

# Train autoencoder
@st.cache_data(show_spinner="Training autoencoder...")
def train_autoencoder(data_values, bottleneck, seed=42):
    """Train an autoencoder using MLPRegressor (input -> bottleneck -> output)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_values)

    # Architecture: input_dim -> 8 -> bottleneck -> 8 -> input_dim
    hidden_layers = (8, bottleneck, 8)

    ae = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        learning_rate_init=0.001,
    )
    ae.fit(X_scaled, X_scaled)  # Target = Input (autoencoder)

    # Compute reconstruction
    X_reconstructed = ae.predict(X_scaled)
    reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

    return scaler, ae, X_scaled, X_reconstructed, reconstruction_error

scaler, ae_model, X_scaled, X_recon, recon_error = train_autoencoder(
    city_data[FEATURE_COLS].values, bottleneck_size
)

# Set threshold
threshold = np.percentile(recon_error, threshold_pctile)
city_data["recon_error"] = recon_error
city_data["ae_anomaly"] = recon_error > threshold

n_ae_anomalies = city_data["ae_anomaly"].sum()

with col_viz:
    fig = go.Figure()

    normal_mask = ~city_data["ae_anomaly"]
    anom_mask = city_data["ae_anomaly"]

    fig.add_trace(go.Scatter(
        x=city_data.loc[normal_mask, x_feat],
        y=city_data.loc[normal_mask, y_feat],
        mode="markers",
        marker=dict(
            color=city_data.loc[normal_mask, "recon_error"],
            colorscale="Viridis",
            size=3, opacity=0.4,
            colorbar=dict(title="Recon Error", x=1.02),
        ),
        name="Normal",
    ))

    fig.add_trace(go.Scatter(
        x=city_data.loc[anom_mask, x_feat],
        y=city_data.loc[anom_mask, y_feat],
        mode="markers",
        marker=dict(color="#E63946", size=8, symbol="x", line=dict(width=1)),
        name=f"Anomaly ({n_ae_anomalies})",
    ))

    fig.update_layout(
        xaxis_title=FEATURE_LABELS.get(x_feat, x_feat),
        yaxis_title=FEATURE_LABELS.get(y_feat, y_feat),
    )
    apply_common_layout(
        fig,
        title=f"Autoencoder Anomalies: {FEATURE_LABELS[x_feat]} vs {FEATURE_LABELS[y_feat]}",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Points", f"{len(city_data):,}")
m2.metric("Anomalies", f"{n_ae_anomalies:,}")
m3.metric("Threshold (RE)", f"{threshold:.4f}")
m4.metric("Bottleneck Dim", f"{bottleneck_size}")

st.divider()

# ---------------------------------------------------------------------------
# 3. Reconstruction Error Distribution
# ---------------------------------------------------------------------------
st.subheader("Reconstruction Error Distribution")

fig_re = go.Figure()
fig_re.add_trace(go.Histogram(
    x=recon_error, nbinsx=100,
    marker_color="#2A9D8F", opacity=0.7,
    name="All Points",
))
fig_re.add_vline(x=threshold, line_dash="dash", line_color="#E63946", line_width=2,
                 annotation_text=f"Threshold ({threshold_pctile}th pctile)")

fig_re.update_layout(
    xaxis_title="Reconstruction Error (MSE)",
    yaxis_title="Count",
)
apply_common_layout(fig_re, title="Reconstruction Error Distribution", height=400)
st.plotly_chart(fig_re, use_container_width=True)

# Per-feature reconstruction error
st.markdown("#### Per-Feature Reconstruction Error")

per_feat_error = np.mean((X_scaled - X_recon) ** 2, axis=0)
feat_error_df = pd.DataFrame({
    "Feature": [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
    "Mean Reconstruction Error": per_feat_error,
}).sort_values("Mean Reconstruction Error", ascending=False)

fig_feat = go.Figure(go.Bar(
    x=feat_error_df["Feature"],
    y=feat_error_df["Mean Reconstruction Error"],
    marker_color=["#E63946", "#F4A261", "#2A9D8F", "#264653"][:len(FEATURE_COLS)],
))
fig_feat.update_layout(xaxis_title="Feature", yaxis_title="Mean Recon Error")
apply_common_layout(fig_feat, title="Which Features Are Hardest to Reconstruct?", height=350)
st.plotly_chart(fig_feat, use_container_width=True)

insight_box(
    "Features with higher reconstruction error are harder for the autoencoder to model -- "
    "they have more complex or variable patterns that resist compression through the bottleneck. "
    "This is useful diagnostic information: it tells you which aspects of weather are most "
    "unpredictable, at least from the perspective of a neural network trying to summarize them "
    "with two numbers."
)

st.divider()

# ---------------------------------------------------------------------------
# 4. Bottleneck Size Sensitivity
# ---------------------------------------------------------------------------
st.subheader("Bottleneck Size Sensitivity")

st.markdown(
    "The bottleneck dimension is the architectural choice that matters most. "
    "A bottleneck of 1 means the autoencoder must compress 4 weather features into a single "
    "number -- extreme compression that creates a very strict definition of 'normal.' "
    "A bottleneck of 3 allows a richer representation and more forgiving reconstruction."
)

bn_results = []
for bn in [1, 2, 3]:
    _, _, _, _, re = train_autoencoder(city_data[FEATURE_COLS].values, bn)
    thresh = np.percentile(re, threshold_pctile)
    n_anom = (re > thresh).sum()
    bn_results.append({
        "Bottleneck Size": bn,
        "Mean Recon Error": f"{re.mean():.4f}",
        "Max Recon Error": f"{re.max():.4f}",
        "Threshold": f"{thresh:.4f}",
        "Anomalies": n_anom,
    })

st.dataframe(pd.DataFrame(bn_results), use_container_width=True, hide_index=True)

insight_box(
    "Bottleneck=1 forces the most extreme compression -- the autoencoder must represent "
    "4 weather features with a single number, which is like trying to describe a symphony "
    "with one adjective. Reconstruction gets harder overall, but the anomalies stick out more "
    "because they deviate from the learned manifold of normality."
)

st.divider()

# ---------------------------------------------------------------------------
# 5. Three-Method Comparison
# ---------------------------------------------------------------------------
st.subheader("Comparison: Autoencoder vs Isolation Forest vs Z-Score")

st.markdown(
    "Now for the grand comparison. We have three fundamentally different approaches to "
    "anomaly detection: a statistical method (Z-score), a tree-based method (Isolation Forest), "
    "and a neural method (autoencoder). Where they agree, we can be quite confident. "
    "Where they disagree, we learn something about the strengths and blind spots of each."
)

# Z-score anomalies (|z| > 3 on any feature)
z_anomaly = np.zeros(len(city_data), dtype=bool)
for feat in FEATURE_COLS:
    vals = city_data[feat].values
    mu, sigma = vals.mean(), vals.std()
    z = np.abs((vals - mu) / (sigma if sigma > 0 else 1))
    z_anomaly |= z > 3

# Isolation Forest
@st.cache_data(show_spinner="Running Isolation Forest...")
def run_iforest(data_values, seed=42):
    scaler_if = StandardScaler()
    X_sc = scaler_if.fit_transform(data_values)
    iso = IsolationForest(contamination=0.03, n_estimators=100, random_state=seed)
    return iso.fit_predict(X_sc) == -1

if_anomaly = run_iforest(city_data[FEATURE_COLS].values)

city_data["z_anomaly"] = z_anomaly
city_data["if_anomaly"] = if_anomaly

# Summary table
methods = {
    "Z-Score (|z|>3)": z_anomaly,
    "Isolation Forest (3%)": if_anomaly,
    f"Autoencoder ({threshold_pctile}th pctile)": city_data["ae_anomaly"].values,
}

summary_rows = []
method_arrays = list(methods.values())
method_names = list(methods.keys())

for name, arr in methods.items():
    summary_rows.append({
        "Method": name,
        "Anomalies": arr.sum(),
        "Rate": f"{arr.mean() * 100:.2f}%",
    })
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# Overlap analysis
all_three = z_anomaly & if_anomaly & city_data["ae_anomaly"].values
any_two = (
    (z_anomaly & if_anomaly) |
    (z_anomaly & city_data["ae_anomaly"].values) |
    (if_anomaly & city_data["ae_anomaly"].values)
)
any_method = z_anomaly | if_anomaly | city_data["ae_anomaly"].values

overlap_rows = [
    {"Category": "Flagged by all 3 methods", "Count": all_three.sum()},
    {"Category": "Flagged by at least 2 methods", "Count": any_two.sum()},
    {"Category": "Flagged by at least 1 method", "Count": any_method.sum()},
    {"Category": "Not flagged by any method", "Count": (~any_method).sum()},
]
st.dataframe(pd.DataFrame(overlap_rows), use_container_width=True, hide_index=True)

# Visual comparison
fig_comp = make_subplots(
    rows=1, cols=3,
    subplot_titles=["Z-Score", "Isolation Forest", "Autoencoder"],
)

for i, (name, arr) in enumerate(methods.items()):
    col = i + 1
    normal_m = ~arr
    fig_comp.add_trace(
        go.Scatter(
            x=city_data.loc[normal_m, x_feat],
            y=city_data.loc[normal_m, y_feat],
            mode="markers", marker=dict(color="#ccc", size=2, opacity=0.2),
            showlegend=False,
        ),
        row=1, col=col,
    )
    fig_comp.add_trace(
        go.Scatter(
            x=city_data.loc[arr, x_feat],
            y=city_data.loc[arr, y_feat],
            mode="markers", marker=dict(color="#E63946", size=5, symbol="x"),
            showlegend=False,
        ),
        row=1, col=col,
    )
    fig_comp.update_xaxes(title_text=FEATURE_LABELS.get(x_feat, x_feat), row=1, col=col)
    fig_comp.update_yaxes(title_text=FEATURE_LABELS.get(y_feat, y_feat), row=1, col=col)

fig_comp.update_layout(template="plotly_white", height=400, margin=dict(t=40, b=40))
st.plotly_chart(fig_comp, use_container_width=True)

# Consensus anomalies
st.markdown("#### High-Confidence Anomalies (Flagged by 2+ Methods)")
consensus = city_data[any_two].copy()
if len(consensus) > 0:
    display_cols = ["datetime"] + FEATURE_COLS
    st.dataframe(
        consensus[display_cols].head(20).reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )
    insight_box(
        f"**{all_three.sum()}** points are flagged by all three methods -- these are the "
        "most reliably anomalous observations in the dataset. Each method has different "
        "assumptions and blind spots, so when a statistical test, a tree ensemble, and a "
        "neural network all independently agree that something is weird, it probably is."
    )
else:
    st.info("No consensus anomalies with current settings.")

st.divider()

# ---------------------------------------------------------------------------
# 6. Code Example
# ---------------------------------------------------------------------------
code_example("""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

features = ['temperature_c', 'relative_humidity_pct',
            'wind_speed_kmh', 'surface_pressure_hpa']
X = city_df[features].dropna().values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train autoencoder (input dim -> 8 -> 2 -> 8 -> input dim)
ae = MLPRegressor(
    hidden_layer_sizes=(8, 2, 8),  # bottleneck = 2
    activation='relu',
    max_iter=300,
    random_state=42,
    early_stopping=True,
)
ae.fit(X_scaled, X_scaled)  # target = input

# Compute reconstruction error
X_recon = ae.predict(X_scaled)
recon_error = np.mean((X_scaled - X_recon) ** 2, axis=1)

# Flag anomalies (top 3% reconstruction error)
threshold = np.percentile(recon_error, 97)
anomalies = recon_error > threshold

print(f"Anomalies: {anomalies.sum()} / {len(X)}")
print(f"Threshold: {threshold:.4f}")

# For production, use PyTorch:
# import torch.nn as nn
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
#         self.decoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 4))
#     def forward(self, x):
#         return self.decoder(self.encoder(x))
""")

st.divider()

# ---------------------------------------------------------------------------
# 7. Quiz
# ---------------------------------------------------------------------------
quiz(
    "Why does an autoencoder detect anomalies through reconstruction error?",
    [
        "Anomalies are always in the training data so the model memorises them",
        "The autoencoder is trained on normal data, so it reconstructs anomalies poorly",
        "The bottleneck stores anomaly labels",
        "The decoder specifically learns to flag anomalies",
    ],
    correct_idx=1,
    explanation="The autoencoder learns to compress and reconstruct normal patterns. "
                "When an anomaly arrives -- a pattern the network has never learned to represent "
                "-- the reconstruction is poor, producing high error. The anomaly 'breaks' the "
                "model, and that breakage is the signal.",
    key="ch59_quiz1",
)

quiz(
    "Which statement about the bottleneck dimension is correct?",
    [
        "Larger bottleneck always detects more anomalies",
        "Bottleneck size has no effect on anomaly detection",
        "Smaller bottleneck forces more compression, creating a stricter normality model",
        "Bottleneck must equal the number of input features",
    ],
    correct_idx=2,
    explanation="A smaller bottleneck forces the autoencoder to learn a more compressed "
                "representation. This makes it harder to reconstruct unusual patterns, "
                "effectively raising the bar for what counts as 'normal.'",
    key="ch59_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 8. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Autoencoders learn to compress and reconstruct normal data patterns. Anomalies are the things they fail to copy.",
    "Reconstruction error is the anomaly score: the worse the reconstruction, the more anomalous the point. Simple, interpretable, effective.",
    "The bottleneck size controls the compression level and anomaly sensitivity -- smaller means stricter, not necessarily better.",
    "The threshold on reconstruction error determines the anomaly rate. This is where your domain judgment enters the equation.",
    "Combining autoencoder, Isolation Forest, and Z-score provides robust consensus anomaly detection. When three different paradigms agree, pay attention.",
])

navigation(
    prev_label="Ch 58: Isolation Forest",
    prev_page="58_Isolation_Forest.py",
    next_label="Ch 60: Correlation vs Causation",
    next_page="60_Correlation_vs_Causation.py",
)
