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
df = load_data()
fdf = sidebar_filters(df)

chapter_header(59, "Autoencoder Anomaly Detection", part="XIV")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
st.markdown(
    "We now have two anomaly detection methods: Z-score (one feature at a time) and "
    "Isolation Forest (all features, but using random splits). Both are powerful, but "
    "neither one *learns* what normal weather actually looks like. Z-score just computes "
    "a mean and standard deviation. Isolation Forest just measures how easy it is to "
    "isolate a point. Neither builds an internal model of 'this is what Houston weather "
    "is supposed to be.'"
)
st.markdown(
    "What if we trained a neural network to do exactly that -- learn a compressed "
    "representation of what normal weather looks like? Then when an abnormal reading "
    "comes in, the network fails to reconstruct it, and that failure *is* the anomaly "
    "signal."
)
st.markdown(
    "That is the autoencoder approach, and the intuition is surprisingly simple."
)

concept_box(
    "The Photocopy Machine Analogy",
    "Imagine you build a photocopier that is trained exclusively on pictures of cats. "
    "You feed it a cat photo, it makes a copy, the copy looks great. Feed it another cat "
    "photo -- another good copy. The copier has learned the essential 'cat-ness' of the images "
    "and can reproduce them faithfully.<br><br>"
    "Now feed it a picture of a dog. The copier tries to reproduce it, but it only knows how "
    "to make cats. The copy comes out looking like a blurry, vaguely cat-shaped mess. "
    "The <b>reconstruction error</b> -- the difference between the original dog picture and "
    "the mangled copy -- is enormous. That error is your anomaly signal.<br><br>"
    "An autoencoder does exactly this with weather data. We train it on normal hourly "
    "readings: temperature, humidity, wind speed, pressure. It learns to compress these "
    "4 numbers into 2 (or 1, or 3) internal numbers and then expand them back to 4. "
    "For a typical Houston reading (30 degrees C, 75% humidity, 8 km/h wind, 1014 hPa), "
    "the reconstruction is nearly perfect -- error of maybe 0.01.<br><br>"
    "But for that anomalous Houston reading (30 degrees C, 20% humidity, 8 km/h, 1014 hPa), "
    "the network tries to reconstruct it and says: 'Houston at 30 degrees C? Humidity should "
    "be around 72%.' It outputs (30, 72, 8, 1014) when the actual input was (30, 20, 8, 1014). "
    "The humidity reconstruction error alone is (72 - 20)^2 = 2,704. The network is essentially "
    "telling us: 'I do not know how to make this pattern, because I have never seen Houston be "
    "this dry at this temperature.'",
)

st.markdown(
    "The key architectural choice is the **bottleneck**. Our data has 4 features. If the "
    "hidden layer also has 4 neurons, the network can just pass everything through unchanged "
    "-- no compression, no learning, and every point reconstructs perfectly (including anomalies). "
    "By forcing the data through a bottleneck of, say, 2 neurons, the network must learn what "
    "matters and throw away the rest. Normal patterns survive the compression. Anomalous "
    "patterns do not."
)

formula_box(
    "Reconstruction Error (Anomaly Score)",
    r"\text{RE}(x) = \|x - \hat{x}\|^2 = \sum_{i=1}^{d} (x_i - \hat{x}_i)^2",
    "x is the original 4-feature input (temp, humidity, wind, pressure), x-hat is the "
    "autoencoder's reconstruction. For that anomalous Houston reading, RE might be 2,710 "
    "(dominated by the humidity mismatch). For a normal reading, RE might be 0.02. "
    "Points with RE above a threshold get flagged as anomalies.",
)

warning_box(
    "We use sklearn's MLPRegressor as a simple autoencoder (training target = input, so "
    "the network learns to copy its input through a bottleneck). For production anomaly "
    "detection, you would want PyTorch or TensorFlow with more control over architecture, "
    "regularization, and training. But the principle is identical: train on normal data, "
    "flag high reconstruction error."
)

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Autoencoder Training & Anomaly Detection
# ---------------------------------------------------------------------------
st.subheader("Interactive: Autoencoder Anomaly Detection")

st.markdown(
    "Here is what each control does and what to watch for:"
)
st.markdown(
    "- **City**: The autoencoder learns what normal weather looks like for this city. "
    "LA has a very narrow definition of 'normal' (mild, dry, stable); Houston has a wider "
    "one (hot and humid, but with more variance).\n"
    "- **Bottleneck size**: The most important architectural choice. Size 1 = compress 4 "
    "features into a single number (extreme compression, very strict 'normal'). Size 3 = "
    "compress 4 into 3 (mild compression, forgiving 'normal'). Try each and watch how the "
    "reconstruction error changes.\n"
    "- **Anomaly threshold (percentile)**: Points above this percentile of reconstruction "
    "error are flagged. At 97th percentile, the top 3% of worst-reconstructed points are "
    "anomalies. Lower = more anomalies; higher = fewer but more confident.\n"
    "- **Scatter axes**: Which two features to visualize. The color gradient shows "
    "reconstruction error -- darker points are harder for the autoencoder to reproduce."
)

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

st.markdown(
    "This histogram shows the reconstruction error for every data point. Most readings "
    "cluster near zero -- the autoencoder reproduces them faithfully because they match "
    "the patterns it learned during training. The long right tail contains the anomalies: "
    "readings the autoencoder struggled to reconstruct because they do not fit its learned "
    "model of 'normal weather in this city.'"
)

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

st.markdown(
    "Which of the 4 weather features is hardest for the autoencoder to reconstruct? "
    "Features with higher error have more complex or variable patterns that resist "
    "compression through the bottleneck. Think of it this way: if wind speed is hardest "
    "to reconstruct, it means wind is the most unpredictable feature from the perspective "
    "of a network that only has 2 internal neurons to work with."
)

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
    "The feature with the highest reconstruction error is the one the autoencoder finds "
    "most 'surprising' -- it carries the most information that cannot be predicted from "
    "the other features. For many cities, wind speed is hardest to reconstruct because "
    "it is the most chaotic feature (temperature and humidity follow smooth seasonal "
    "patterns, but wind can spike or drop unpredictably). Surface pressure, on the other "
    "hand, is often easiest to reconstruct because it is relatively stable. This ranking "
    "tells you which aspects of weather are most 'compressible' by a neural network."
)

st.divider()

# ---------------------------------------------------------------------------
# 4. Bottleneck Size Sensitivity
# ---------------------------------------------------------------------------
st.subheader("Bottleneck Size Sensitivity")

st.markdown(
    "The bottleneck dimension is the architectural choice that matters most, so let me "
    "explain what each option means concretely. Our input is 4 numbers: temperature, "
    "humidity, wind, pressure."
)
st.markdown(
    "- **Bottleneck = 1**: The autoencoder must compress all 4 features into a single "
    "number. That is like describing a 4-dimensional weather state with one adjective. "
    "It might capture 'hot-and-humid vs cold-and-dry' but nothing more nuanced. "
    "Reconstruction error is high for everyone, but anomalies stick out more.\n"
    "- **Bottleneck = 2**: Two internal numbers. The network can represent something like "
    "'temperature-humidity axis' and 'wind-pressure axis.' Much better compression, and "
    "the most common choice for 4-feature data.\n"
    "- **Bottleneck = 3**: Three numbers for 4 features -- very mild compression. Almost "
    "everything reconstructs well, so only the most extreme anomalies have high error."
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
    "Notice how the mean reconstruction error drops as the bottleneck gets larger -- "
    "more internal neurons means better reconstruction for everyone. But the anomaly "
    "count stays roughly the same (because we are using a percentile threshold, which "
    "always flags the top N%). The real difference is in what gets flagged: with bottleneck=1, "
    "the autoencoder has such a crude model of normality that even mildly unusual readings "
    "look anomalous. With bottleneck=3, only truly extreme combinations survive as anomalies "
    "because the network can represent most weather patterns accurately."
)

st.divider()

# ---------------------------------------------------------------------------
# 5. Three-Method Comparison
# ---------------------------------------------------------------------------
st.subheader("Comparison: Autoencoder vs Isolation Forest vs Z-Score")

st.markdown(
    "Now for the grand comparison. We have three fundamentally different approaches to "
    "the same question ('is this weather reading unusual?'): a statistical method (Z-score), "
    "a tree-based method (Isolation Forest), and a neural method (autoencoder). Each has "
    "different assumptions and blind spots. Where they agree, we can be quite confident. "
    "Where they disagree, we learn something about each method's strengths."
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

st.markdown(
    "These are the readings where at least two of our three methods independently agree "
    "that something is unusual. A statistical test, a tree ensemble, and a neural network "
    "all have different assumptions -- so when two or more converge on the same point, "
    "there is a genuine signal."
)

consensus = city_data[any_two].copy()
if len(consensus) > 0:
    display_cols = ["datetime"] + FEATURE_COLS
    st.dataframe(
        consensus[display_cols].head(20).reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )
    insight_box(
        f"**{all_three.sum()}** points are flagged by all three methods -- these are the "
        "highest-confidence anomalies in the dataset. Look at the actual values in the table "
        "above. Do they look unusual for this city? You might see extreme temperature-humidity "
        "combinations, or readings where wind speed and pressure suggest storm conditions. "
        "The fact that three completely different algorithms (one statistical, one tree-based, "
        "one neural) all independently flag the same readings is strong evidence that these "
        "are genuinely unusual weather events, not statistical artifacts."
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
    "The autoencoder is trained on normal Houston weather data. A new reading comes in: "
    "temperature = 30 degrees C, humidity = 20%, wind = 8 km/h, pressure = 1013 hPa. "
    "The autoencoder reconstructs it as (30, 71, 9, 1013). Why is the reconstruction "
    "error high?",
    [
        "The autoencoder memorized this exact reading and is reproducing it from memory",
        "The autoencoder learned that Houston at 30 degrees C normally has ~71% humidity, "
        "so it reconstructs what it *expects* rather than the actual input of 20%",
        "The bottleneck is too large and the network cannot compress the data",
        "The decoder specifically learned to flag anomalies by outputting wrong values",
    ],
    correct_idx=1,
    explanation="The autoencoder has learned the statistical structure of normal Houston weather. "
                "It knows that when temperature is 30 degrees C, humidity is typically around "
                "65-75%. When it receives an input with 20% humidity, it cannot represent "
                "that unusual combination through its compressed bottleneck -- so it outputs "
                "what it *expects* (71%) rather than what it received (20%). The squared error "
                "on humidity alone is (71 - 20)^2 = 2,601, which dominates the reconstruction "
                "error. This is exactly the mechanism: the network fails to copy unusual patterns "
                "because they do not fit the compressed representation of normality it learned.",
    key="ch59_quiz1",
)

quiz(
    "You change the bottleneck from size 2 to size 1. What happens to the reconstruction "
    "error and anomaly detection?",
    [
        "Reconstruction error decreases because the network is simpler",
        "Reconstruction error increases for everyone because the compression is more extreme, "
        "but anomalies stick out more relative to the higher baseline",
        "Nothing changes because the bottleneck size does not affect reconstruction",
        "The network can no longer train because 1 neuron is too few",
    ],
    correct_idx=1,
    explanation="With bottleneck=1, the network must squeeze 4 weather features into a single "
                "number. That is like trying to describe temperature, humidity, wind, and pressure "
                "with one word. Even normal readings get reconstructed imperfectly -- the mean error "
                "goes up for everyone. But anomalous readings get reconstructed even *worse* because "
                "they deviate from the already-strained model of normality. The anomalies stick out "
                "more against the higher baseline, making detection more sensitive but also more "
                "prone to false positives on mildly unusual (but not truly anomalous) readings.",
    key="ch59_quiz2",
)

quiz(
    "Z-score flags a Dallas reading with wind speed = 78 km/h as anomalous (|z| = 4.2). "
    "The autoencoder does NOT flag it. Which explanation is most plausible?",
    [
        "The autoencoder is wrong and should be retrained",
        "The autoencoder learned that high wind often co-occurs with low pressure and low "
        "temperature (storm conditions), so it can reconstruct this reading accurately",
        "Z-score is always more reliable than autoencoders",
        "The bottleneck is too small to capture wind patterns",
    ],
    correct_idx=1,
    explanation="This is a beautiful example of the difference between univariate and "
                "learned-model approaches. Z-score sees 78 km/h wind and compares it against "
                "the marginal wind distribution -- it is extreme, so it flags it. But the "
                "autoencoder has learned the *correlations* between features. It knows that "
                "when pressure drops to 990 hPa and temperature drops to 5 degrees C (storm "
                "conditions), high winds are expected. So it reconstructs the reading accurately: "
                "'given the pressure and temperature, 78 km/h wind makes sense.' The error is low, "
                "and the point is not flagged. The autoencoder is not wrong -- it is using richer "
                "information than Z-score has access to.",
    key="ch59_quiz3",
)

st.divider()

# ---------------------------------------------------------------------------
# 8. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "An autoencoder learns to compress and reconstruct normal weather patterns. When an "
    "anomalous reading arrives -- like 20% humidity in Houston at 30 degrees C -- the network "
    "fails to reconstruct it because that pattern was never part of its training. The "
    "reconstruction error IS the anomaly score.",
    "The bottleneck size controls how strictly the network defines 'normal.' Bottleneck=1 "
    "means the network summarizes 4 weather features with 1 number (extreme compression, "
    "sensitive but noisy). Bottleneck=3 means 4 features compressed to 3 (mild compression, "
    "only catches the most extreme anomalies).",
    "The per-feature reconstruction error tells you which aspects of weather are hardest "
    "to predict. Wind speed is typically the most unpredictable; pressure is the most stable. "
    "This is the network discovering the physics of weather through pure pattern matching.",
    "When Z-score, Isolation Forest, and the autoencoder all agree a reading is anomalous, "
    "you can be very confident -- three completely different paradigms (statistical, tree-based, "
    "neural) have independently converged on the same conclusion.",
    "The threshold on reconstruction error is your domain judgment knob. At the 97th "
    "percentile, you flag 3% of readings. At the 99th, you flag 1%. The right answer "
    "depends on whether you want to catch every mildly unusual event or only the truly extreme ones.",
])

navigation(
    prev_label="Ch 58: Isolation Forest",
    prev_page="58_Isolation_Forest.py",
    next_label="Ch 60: Correlation vs Causation",
    next_page="60_Correlation_vs_Causation.py",
)
