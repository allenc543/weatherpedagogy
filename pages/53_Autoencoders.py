"""Chapter 53: Autoencoders -- Encoder-decoder, latent space, reconstruction error."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(53, "Autoencoders", part="XII")
st.markdown(
    "Autoencoders are one of those ideas that sounds like it should not work. "
    "You take a neural network, you tell it to reconstruct its own input, and "
    "somehow this teaches it something useful about the structure of the data. "
    "It is the machine learning equivalent of asking someone to summarize a "
    "book and then reconstruct the book from the summary -- the interesting "
    "part is what survives the compression. This chapter explores how "
    "autoencoders **compress** weather data into a latent space and what we "
    "can learn from the reconstruction errors."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 53.1 How Autoencoders Work ──────────────────────────────────────────────
st.header("53.1  The Autoencoder Architecture")

concept_box(
    "Encoder-Decoder Structure",
    "An autoencoder has two halves, and they are locked in a productive "
    "adversarial relationship:<br><br>"
    "- <b>Encoder</b>: takes the input X (4 weather features) and squeezes it "
    "through a narrow bottleneck into a compressed representation Z. This is "
    "the \"summarize the book\" part.<br>"
    "- <b>Decoder</b>: takes Z and tries to reconstruct the original input X. "
    "This is the \"reconstruct the book from the summary\" part.<br><br>"
    "The <b>bottleneck</b> is what makes it interesting. By forcing 4 dimensions "
    "through, say, 2 dimensions, the network must learn which information is "
    "essential and which can be sacrificed. The loss function is just "
    "<b>reconstruction error</b>: how close is the output to the original "
    "input? That is it. No labels required -- this is unsupervised learning."
)

formula_box(
    "Autoencoder Objective",
    r"\min_{\theta, \phi} \frac{1}{n}\sum_{i=1}^{n}\|x_i - D_\phi(E_\theta(x_i))\|^2",
    "E = encoder, D = decoder, theta/phi = their respective parameters. "
    "We are literally minimizing \"how different is the output from the input "
    "after a round trip through the bottleneck.\""
)

st.markdown("""
**Our weather autoencoder architecture:**
```
Input (4 features) --> Encoder --> Bottleneck (2D latent space) --> Decoder --> Output (4 features)
```

The 4 features are: temperature, humidity, wind speed, and surface pressure. We are asking the network: "Can you describe a weather observation using only 2 numbers instead of 4, and then reconstruct the original 4 numbers from those 2?" You might ask why this is useful. Two reasons: (1) it tells us which features carry redundant information, and (2) the latent space gives us a 2D map of weather states that we can actually visualize.
""")

# ── 53.2 Prepare Data ───────────────────────────────────────────────────────
clean = fdf[FEATURE_COLS + ["city"]].dropna()
sample_n = min(10000, len(clean))
sample_data = clean.sample(sample_n, random_state=42)

X_all = sample_data[FEATURE_COLS].values
cities = sample_data["city"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

X_train, X_test, cities_train, cities_test = train_test_split(
    X_scaled, cities, test_size=0.2, random_state=42
)

# ── 53.3 Interactive Bottleneck Size ─────────────────────────────────────────
st.header("53.2  Interactive: Bottleneck Size Selector")

st.markdown(
    "The bottleneck size is the single most important design choice in an "
    "autoencoder. With 4 input features, we can compress to 1, 2, or 3 "
    "dimensions. A bottleneck of 4 would be pointless -- that is just copying. "
    "A bottleneck of 1 is aggressive -- can you really describe weather with "
    "a single number? Let us find out."
)

bottleneck_size = st.slider("Bottleneck dimensions (latent space size)", 1, 3, 2, key="bn_size")

# Build autoencoder using two MLPRegressors (encoder + decoder pattern via single MLP)
# sklearn MLP autoencoder: input=4 -> hidden=encoder -> bottleneck -> hidden=decoder -> output=4
# Architecture: 4 -> 8 -> bottleneck -> 8 -> 4

st.markdown(f"""
**Current architecture: 4 --> 8 --> {bottleneck_size} --> 8 --> 4**
""")

with st.spinner("Training autoencoder..."):
    ae = MLPRegressor(
        hidden_layer_sizes=(8, bottleneck_size, 8),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        learning_rate_init=0.005,
    )
    ae.fit(X_train, X_train)  # Target = Input (reconstruction)

X_train_recon = ae.predict(X_train)
X_test_recon = ae.predict(X_test)

train_mse = mean_squared_error(X_train, X_train_recon)
test_mse = mean_squared_error(X_test, X_test_recon)

col1, col2, col3 = st.columns(3)
col1.metric("Bottleneck Size", f"{bottleneck_size}D")
col2.metric("Train Reconstruction MSE", f"{train_mse:.4f}")
col3.metric("Test Reconstruction MSE", f"{test_mse:.4f}")

# Per-feature reconstruction error
feature_mses_train = []
feature_mses_test = []
for i, feat in enumerate(FEATURE_COLS):
    feature_mses_train.append(mean_squared_error(X_train[:, i], X_train_recon[:, i]))
    feature_mses_test.append(mean_squared_error(X_test[:, i], X_test_recon[:, i]))

feat_error_df = pd.DataFrame({
    "Feature": [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
    "Train MSE": feature_mses_train,
    "Test MSE": feature_mses_test,
})
st.dataframe(
    feat_error_df.style.format({"Train MSE": "{:.4f}", "Test MSE": "{:.4f}"}),
    use_container_width=True, hide_index=True,
)

# Reconstruction comparison: original vs reconstructed for a few samples
st.subheader("Original vs Reconstructed (Test Set Samples)")
n_show = min(8, len(X_test))
orig_unscaled = scaler.inverse_transform(X_test[:n_show])
recon_unscaled = scaler.inverse_transform(X_test_recon[:n_show])

for i in range(min(3, n_show)):
    col_a, col_b = st.columns(2)
    with col_a:
        sample_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
            "Original": orig_unscaled[i],
            "Reconstructed": recon_unscaled[i],
            "Error": np.abs(orig_unscaled[i] - recon_unscaled[i]),
        })
        st.markdown(f"**Sample {i+1}** ({cities_test[i]})")
        st.dataframe(sample_df.style.format({"Original": "{:.2f}", "Reconstructed": "{:.2f}", "Error": "{:.2f}"}),
                     use_container_width=True, hide_index=True)

# ── 53.4 Reconstruction error vs bottleneck size ────────────────────────────
st.header("53.3  Reconstruction Error vs Compression Level")

bn_sizes = [1, 2, 3]
bn_mses_train = []
bn_mses_test = []

for bn in bn_sizes:
    ae_temp = MLPRegressor(
        hidden_layer_sizes=(8, bn, 8),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        learning_rate_init=0.005,
    )
    ae_temp.fit(X_train, X_train)
    bn_mses_train.append(mean_squared_error(X_train, ae_temp.predict(X_train)))
    bn_mses_test.append(mean_squared_error(X_test, ae_temp.predict(X_test)))

bn_df = pd.DataFrame({
    "Bottleneck Size": bn_sizes,
    "Train MSE": bn_mses_train,
    "Test MSE": bn_mses_test,
    "Compression Ratio": [f"4:{bn}" for bn in bn_sizes],
})

fig_bn = go.Figure()
fig_bn.add_trace(go.Bar(x=[str(b) for b in bn_sizes], y=bn_mses_train,
                         name="Train MSE", marker_color="#2A9D8F"))
fig_bn.add_trace(go.Bar(x=[str(b) for b in bn_sizes], y=bn_mses_test,
                         name="Test MSE", marker_color="#E63946"))
apply_common_layout(fig_bn, title="Reconstruction Error vs Bottleneck Size", height=400)
fig_bn.update_layout(barmode="group", xaxis_title="Bottleneck Dimensions", yaxis_title="MSE")
st.plotly_chart(fig_bn, use_container_width=True)

st.dataframe(bn_df.style.format({"Train MSE": "{:.4f}", "Test MSE": "{:.4f}"}),
             use_container_width=True, hide_index=True)

insight_box(
    "There is an obvious trade-off here: larger bottlenecks preserve more "
    "information (lower reconstruction error) but provide less compression. "
    "The 4-to-2-to-4 architecture turns out to be a sweet spot -- 50% "
    "compression while keeping reconstruction quality reasonable. Going down "
    "to 1 dimension is brutal; going up to 3 barely compresses at all. This "
    "mirrors what we see in PCA: the first two principal components usually "
    "capture most of the variance in weather data, because temperature and "
    "pressure dominate the show."
)

# ── 53.5 Latent Space Visualization ─────────────────────────────────────────
st.header("53.4  Latent Space Visualization: Autoencoder vs PCA")

concept_box(
    "Latent Space",
    "The bottleneck layer's activations form the <b>latent space</b> -- a "
    "compressed coordinate system for weather observations. If the bottleneck "
    "is 2D, we can directly plot it and see how the autoencoder has organized "
    "the data. The interesting question is: does the network discover "
    "something like geography? Do cities cluster together? And how does this "
    "non-linear projection compare to PCA's linear one?"
)

# PCA 2D projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Autoencoder 2D latent space (train a 4->8->2->8->4 AE)
ae_2d = MLPRegressor(
    hidden_layer_sizes=(8, 2, 8),
    activation="relu",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    learning_rate_init=0.005,
)
ae_2d.fit(X_scaled, X_scaled)

# Extract latent representations by creating an encoder
# We use a trick: train a separate encoder network to predict the bottleneck
# Instead, we use the intermediate layer weights to project manually
# For sklearn MLP, we access weights: coefs_[0] (input->h1), coefs_[1] (h1->bottleneck)
if len(ae_2d.coefs_) >= 2:
    # Manual forward pass through encoder layers
    h1 = np.maximum(0, X_scaled @ ae_2d.coefs_[0] + ae_2d.intercepts_[0])  # ReLU
    latent = np.maximum(0, h1 @ ae_2d.coefs_[1] + ae_2d.intercepts_[1])  # ReLU
else:
    latent = X_pca  # fallback

# Create DataFrames for plotting
pca_df = pd.DataFrame({
    "Component 1": X_pca[:, 0],
    "Component 2": X_pca[:, 1],
    "City": cities,
})
ae_df = pd.DataFrame({
    "Latent 1": latent[:, 0],
    "Latent 2": latent[:, 1],
    "City": cities,
})

col_pca, col_ae = st.columns(2)

with col_pca:
    fig_pca = px.scatter(pca_df, x="Component 1", y="Component 2", color="City",
                         color_discrete_map=CITY_COLORS, opacity=0.3,
                         title="PCA 2D Projection")
    apply_common_layout(fig_pca, height=450)
    fig_pca.update_traces(marker_size=3)
    st.plotly_chart(fig_pca, use_container_width=True)

with col_ae:
    fig_ae = px.scatter(ae_df, x="Latent 1", y="Latent 2", color="City",
                        color_discrete_map=CITY_COLORS, opacity=0.3,
                        title="Autoencoder Latent Space")
    apply_common_layout(fig_ae, height=450)
    fig_ae.update_traces(marker_size=3)
    st.plotly_chart(fig_ae, use_container_width=True)

# Variance explained comparison
pca_full = PCA(n_components=len(FEATURE_COLS))
pca_full.fit(X_scaled)
pca_var_2d = sum(pca_full.explained_variance_ratio_[:2])
ae_recon_mse = mean_squared_error(X_scaled, ae_2d.predict(X_scaled))
total_var = np.var(X_scaled, axis=0).sum()
ae_var_explained = 1 - ae_recon_mse / total_var

compare_df = pd.DataFrame({
    "Method": ["PCA (2D)", "Autoencoder (2D bottleneck)"],
    "Variance Explained (approx)": [pca_var_2d, ae_var_explained],
    "Type": ["Linear", "Non-linear"],
})
st.dataframe(
    compare_df.style.format({"Variance Explained (approx)": "{:.4f}"}),
    use_container_width=True, hide_index=True,
)

insight_box(
    "PCA finds the best **linear** 2D projection -- it is restricted to "
    "planes and hyperplanes. The autoencoder can learn **non-linear** "
    "mappings, which means it can bend and twist the projection surface to "
    "fit the data better. In practice, this often produces tighter, more "
    "separated city clusters. Whether this matters depends on your downstream "
    "task, but it is a nice demonstration that sometimes a little non-linearity "
    "goes a long way."
)

# ── 53.6 Anomaly Detection with Reconstruction Error ────────────────────────
st.header("53.5  Anomaly Detection via Reconstruction Error")

concept_box(
    "Anomalies and Reconstruction",
    "Here is an elegant trick. An autoencoder trained on normal data learns "
    "the <b>typical patterns</b> in that data. When you feed it something "
    "unusual -- a freak heat wave, an instrument malfunction, a data entry "
    "error -- it cannot reconstruct it well, because it has never learned to "
    "represent that pattern. The reconstruction error spikes. So high "
    "reconstruction error becomes a natural anomaly detector, and you never "
    "had to label a single anomaly to train it. This is unsupervised anomaly "
    "detection, and it is genuinely clever."
)

# Compute per-sample reconstruction error
recon_all = ae_2d.predict(X_scaled)
sample_mses = np.mean((X_scaled - recon_all) ** 2, axis=1)

error_df = pd.DataFrame({
    "Reconstruction MSE": sample_mses,
    "City": cities,
})

# Distribution of reconstruction errors
fig_error_dist = px.histogram(
    error_df, x="Reconstruction MSE", color="City",
    color_discrete_map=CITY_COLORS, nbins=60,
    title="Distribution of Reconstruction Errors by City",
    barmode="overlay", opacity=0.6,
)
apply_common_layout(fig_error_dist, height=400)
st.plotly_chart(fig_error_dist, use_container_width=True)

# Threshold for anomalies
threshold_pct = st.slider("Anomaly threshold (percentile)", 90, 99, 95, key="anomaly_thresh")
threshold_val = np.percentile(sample_mses, threshold_pct)
n_anomalies = (sample_mses > threshold_val).sum()

st.markdown(f"""
- **Threshold (p{threshold_pct}):** {threshold_val:.4f}
- **Flagged anomalies:** {n_anomalies} out of {len(sample_mses)} ({n_anomalies/len(sample_mses)*100:.1f}%)
""")

# Show anomalous cities
anomaly_mask = sample_mses > threshold_val
anomaly_cities = pd.Series(cities[anomaly_mask]).value_counts()
all_cities_count = pd.Series(cities).value_counts()
anomaly_rate = (anomaly_cities / all_cities_count * 100).round(2)

anomaly_rate_df = pd.DataFrame({
    "City": anomaly_rate.index,
    "Anomaly Rate (%)": anomaly_rate.values,
})
st.dataframe(anomaly_rate_df, use_container_width=True, hide_index=True)

insight_box(
    "Notice that the anomaly rates are not uniform across cities. Cities with "
    "more extreme or unusual weather patterns -- New York with its polar "
    "vortex events, Los Angeles with its Santa Ana winds -- tend to have "
    "higher anomaly rates. This is because the autoencoder learns a kind of "
    "\"average weather\" representation, and observations that deviate sharply "
    "from this average look anomalous. Whether this is a feature or a bug "
    "depends on what you are trying to detect."
)

code_example("""import torch
import torch.nn as nn

class WeatherAutoencoder(nn.Module):
    def __init__(self, input_dim=4, bottleneck_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)        # latent representation
        x_hat = self.decoder(z)     # reconstruction
        return x_hat

    def encode(self, x):
        return self.encoder(x)

# Training
model = WeatherAutoencoder(input_dim=4, bottleneck_dim=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    x_hat = model(X_train_tensor)
    loss = criterion(x_hat, X_train_tensor)  # reconstruct input
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Anomaly detection
recon_error = ((model(X_test_tensor) - X_test_tensor) ** 2).mean(dim=1)
anomalies = recon_error > threshold
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "What is the 'bottleneck' in an autoencoder?",
    [
        "The output layer",
        "The narrowest hidden layer that forces compression of the input",
        "The loss function",
        "The learning rate",
    ],
    correct_idx=1,
    explanation=(
        "The bottleneck is the narrow hidden layer between encoder and decoder "
        "-- the eye of the needle that all information must pass through. It is "
        "what forces the network to learn a compressed representation instead of "
        "just memorizing a copy. Without the bottleneck, the network would learn "
        "the identity function and we would have accomplished nothing."
    ),
    key="ch53_quiz1",
)

quiz(
    "How can autoencoders detect anomalies?",
    [
        "By classifying data into normal and abnormal",
        "Normal data has low reconstruction error; anomalous data has high reconstruction error",
        "By counting the number of neurons that activate",
        "By measuring the latent space dimensions",
    ],
    correct_idx=1,
    explanation=(
        "The logic is beautifully simple: the autoencoder learns to reconstruct "
        "normal patterns well. When it encounters something it has never seen "
        "-- an anomaly -- it does a poor job reconstructing it, and the "
        "reconstruction error spikes. No labeled anomalies needed. The only "
        "assumption is that anomalies are rare enough that the autoencoder "
        "does not learn to reconstruct them too."
    ),
    key="ch53_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "An autoencoder learns to **compress** data (encoder) and **reconstruct** it (decoder). The bottleneck is where the learning happens.",
    "The **bottleneck** forces information through a narrow channel, and what survives tells you what the network considers essential.",
    "**Smaller bottlenecks** = more aggressive compression but higher reconstruction error. There is no free lunch, even in unsupervised learning.",
    "The 4-to-2-to-4 architecture compresses weather data to 2D while preserving most of the essential structure -- a surprisingly good deal.",
    "Autoencoder latent spaces are **non-linear** projections, which can separate clusters that PCA smears together.",
    "**High reconstruction error** is a natural anomaly signal: if the network cannot reconstruct an observation, that observation is probably unusual.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 52: RNN & LSTM",
    prev_page="52_RNN_and_LSTM.py",
    next_label="Ch 54",
    next_page="54_Bayesian_Thinking.py",
)
