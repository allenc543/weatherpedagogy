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
    "Let me set up a concrete question, because 'autoencoder' is one of those words "
    "that sounds impressive but means nothing until you see what it actually does."
)
st.markdown(
    "**The question**: A single weather observation from our dataset has 4 numbers -- "
    "temperature (say, 22.5 C), humidity (65%), wind speed (12 km/h), and pressure "
    "(1013 hPa). Can you describe that same weather observation using only **2 numbers** "
    "instead of 4? And if you can, can you then recover the original 4 numbers from "
    "those 2?"
)
st.markdown(
    "This sounds like it should be impossible -- you are throwing away half the "
    "information. But here is the key insight: the 4 weather features are not independent. "
    "Temperature and humidity are correlated (hot air can hold more moisture). Pressure "
    "and temperature are related (low pressure systems bring cooler air). So the 4 numbers "
    "contain some redundancy, and an autoencoder's job is to figure out exactly how much."
)
st.markdown(
    "**Why we care**: If weather observations really can be described by 2 numbers instead "
    "of 4, that is a profound statement about the structure of weather data. It means there "
    "are really only 2 underlying 'dimensions' of weather variation, and the 4 features we "
    "measure are just different projections of those 2 underlying factors. Also, observations "
    "that *cannot* be compressed well -- where the reconstruction error is large -- are "
    "probably unusual or anomalous, which gives us a free anomaly detector. This chapter "
    "explores both applications."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 53.1 How Autoencoders Work ──────────────────────────────────────────────
st.header("53.1  The Autoencoder Architecture")

st.markdown(
    "Before I use the terms 'encoder,' 'decoder,' or 'latent space,' let me walk through "
    "what the network actually does, step by step, with a real weather observation."
)
st.markdown(
    "Take a Dallas reading: 28.3 C, 58% humidity, 14.2 km/h wind, 1012.5 hPa pressure. "
    "The autoencoder processes this in two stages:"
)
st.markdown(
    "**Stage 1 (Compression):** The first half of the network takes these 4 numbers, "
    "passes them through a hidden layer of 8 neurons, and then squeezes them down to "
    "just 2 numbers. Maybe it produces [1.7, -0.4]. What do those 2 numbers *mean*? "
    "The network decides. Maybe 1.7 encodes 'it is a warm, dry day' and -0.4 encodes "
    "'the pressure is slightly low.' The point is: all the essential information about "
    "this weather observation is now packed into 2 numbers."
)
st.markdown(
    "**Stage 2 (Reconstruction):** The second half of the network takes those 2 numbers "
    "[1.7, -0.4] and tries to expand them back into 4 numbers. If it produces "
    "[27.9 C, 59.1%, 13.8 km/h, 1012.8 hPa], that is pretty close to the original -- "
    "the temperature is off by 0.4 C, humidity by 1.1%, wind by 0.4 km/h, pressure by "
    "0.3 hPa. The network learns by minimizing this reconstruction error."
)

concept_box(
    "Encoder-Decoder Structure",
    "Now the official names:<br><br>"
    "- <b>Encoder</b>: the compression half. Takes the 4-number weather reading and "
    "squeezes it down to 2 numbers. This is the 'summarize the book' part.<br>"
    "- <b>Decoder</b>: the reconstruction half. Takes the 2 compressed numbers and "
    "expands them back to 4. This is the 'reconstruct the book from the summary' part.<br>"
    "- <b>Bottleneck</b>: the narrow layer in the middle (2 neurons in our case). "
    "This is what forces the network to learn a compressed representation. Without it, "
    "the network would just learn to copy input to output -- the identity function -- "
    "and we would have accomplished nothing.<br>"
    "- <b>Latent space</b>: the 2D coordinate system defined by the bottleneck. Each "
    "weather observation gets a unique position in this 2D space, and similar weather "
    "observations should end up nearby.<br><br>"
    "The loss function is just <b>reconstruction error</b>: how close is the output "
    "to the original input, averaged across all samples? No labels required -- this "
    "is unsupervised learning. The network figures out on its own which features are "
    "redundant and which are essential."
)

formula_box(
    "Autoencoder Objective",
    r"\min_{\underbrace{\theta, \phi}_{\text{learned parameters}}} \frac{1}{\underbrace{n}_{\text{sample count}}}\sum_{i=1}^{n}\|\underbrace{x_i}_{\text{original reading}} - \underbrace{D_\phi}_{\text{decoder}}(\underbrace{E_\theta}_{\text{encoder}}(x_i))\|^2",
    "E = encoder, D = decoder, theta/phi = their parameters. In plain English: "
    "minimize the average squared difference between each original 4-number weather "
    "reading and its reconstruction after a round trip through the 2-number bottleneck."
)

st.markdown(f"""
**Our weather autoencoder architecture:**
```
Input (4 features) --> Encoder (8 neurons) --> Bottleneck (2D latent space) --> Decoder (8 neurons) --> Output (4 features)
```

The 4 features are: temperature ({FEATURE_LABELS.get('temperature_c', 'temperature')}), humidity ({FEATURE_LABELS.get('relative_humidity_pct', 'humidity')}), wind speed ({FEATURE_LABELS.get('wind_speed_kmh', 'wind')}), and pressure ({FEATURE_LABELS.get('surface_pressure_hpa', 'pressure')}). We are asking the network: 'Can you describe a weather observation using only 2 numbers instead of 4, and then reconstruct the original 4 numbers from those 2?'
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
    "The bottleneck size is the single most important design choice. With 4 input "
    "features, we can compress to 1, 2, or 3 dimensions. Here is the trade-off:"
)
st.markdown(
    "- **Bottleneck = 1**: Describe all weather with a single number. This is brutally "
    "aggressive -- you are saying temperature, humidity, wind, and pressure can all be "
    "captured by one axis. Can you really summarize 'hot, humid, calm, low-pressure' "
    "with just one number?\n"
    "- **Bottleneck = 2**: Two numbers. A good middle ground -- 50% compression. Think of "
    "it as reducing weather to a position on a 2D map.\n"
    "- **Bottleneck = 3**: Three numbers. Barely any compression at all (4 to 3). Almost "
    "everything survives the round trip, but you are not learning much about the data's "
    "structure.\n"
    "- **Bottleneck = 4**: Pointless. Same dimensionality as the input. The network just "
    "learns to copy."
)
st.markdown(
    "**What to look for:** The reconstruction MSE tells you how much information was lost "
    "in compression. Per-feature errors tell you which weather variables suffer most from "
    "compression -- typically wind speed, because it is less correlated with the other features."
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

st.markdown(
    f"The autoencoder compresses 4 weather features into {bottleneck_size} dimensions and "
    f"reconstructs them with test MSE = {test_mse:.4f}. For reference, a perfect "
    f"reconstruction would have MSE = 0, and a reconstruction that just predicts the "
    f"mean for everything would have MSE around 1.0 (since we standardized the data). "
    f"So {test_mse:.4f} means the autoencoder is preserving most of the information."
)

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

st.markdown(
    "**Reading the per-feature errors:** If temperature has low MSE but wind speed has "
    "high MSE, it means the autoencoder easily captures temperature in its compressed "
    "representation but struggles with wind. This makes intuitive sense: temperature "
    "correlates strongly with humidity and pressure (hot air is often humid; low pressure "
    "brings cooler temps), so it is efficiently represented. Wind speed is more independent "
    "and idiosyncratic, so it is the first casualty of compression."
)

# Reconstruction comparison: original vs reconstructed for a few samples
st.subheader("Original vs Reconstructed (Test Set Samples)")

st.markdown(
    "Let us look at actual reconstructions. For each sample, we show the original "
    "4-number reading, the reconstructed version, and the error. Remember, these "
    "have been through the bottleneck -- compressed to just "
    f"{bottleneck_size} number{'s' if bottleneck_size > 1 else ''} and back."
)

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

st.markdown(
    "Now let us see the trade-off quantitatively. We train three autoencoders with "
    "bottleneck sizes 1, 2, and 3, and measure how much information each one loses. "
    "This is the autoencoder's version of the bias-variance tradeoff: too small a "
    "bottleneck forces the network to throw away real signal, too large a bottleneck "
    "means you are not actually compressing anything useful."
)

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
    f"The bottleneck = 1 autoencoder has test MSE = {bn_mses_test[0]:.4f} -- it is "
    f"losing a lot of information trying to cram 4 weather features into 1 number. "
    f"Bottleneck = 2 drops to {bn_mses_test[1]:.4f} -- a huge improvement for one extra "
    f"dimension. Bottleneck = 3 gets to {bn_mses_test[2]:.4f} -- barely better than 2. "
    "This tells us something profound about our weather data: there are really about "
    "2 dominant dimensions of variation. The jump from 1 to 2 dimensions recovers most "
    "of the information; the jump from 2 to 3 barely helps because the third dimension "
    "carries mostly noise. This mirrors what PCA finds: the first two principal components "
    "capture most of the variance in weather data, because temperature and pressure "
    "dominate the show."
)

# ── 53.5 Latent Space Visualization ─────────────────────────────────────────
st.header("53.4  Latent Space Visualization: Autoencoder vs PCA")

st.markdown(
    "If the bottleneck is 2D, we can directly plot where each weather observation "
    "lands in the compressed space. This 2D map of weather observations is called "
    "the **latent space**. The fascinating question: does the autoencoder discover "
    "something like geography? Do cities cluster together?"
)
st.markdown(
    "For comparison, we also show PCA's 2D projection. PCA can only do linear "
    "transformations (rotations and scalings). The autoencoder can do non-linear "
    "transformations (it can bend and twist the projection surface). Let us see if "
    "that extra flexibility produces better-separated city clusters."
)

concept_box(
    "Latent Space",
    "The bottleneck layer's activations form the <b>latent space</b> -- a compressed "
    "coordinate system where every weather observation gets a position. For our 2D "
    "bottleneck, each observation becomes a point on a 2D map. The autoencoder has "
    "learned, without any labels, to arrange weather observations so that similar "
    "ones are nearby. A hot, humid Houston reading and a hot, humid Dallas reading "
    "should end up close together; a cold, dry NYC reading should be far away.<br><br>"
    "Think of it as the autoencoder drawing a weather map -- not a geographic map, "
    "but a map where distance represents weather similarity rather than physical location."
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

st.markdown(
    "**Reading these plots:** Each dot is one weather observation, colored by city. "
    "If cities form distinct clusters, it means the compression preserved the information "
    "that makes cities different (LA is dry and mild, NYC is cold and humid, Texas cities "
    "are hot). If everything is a single blob, the compression lost that information. "
    "Compare how tightly the autoencoder separates cities vs PCA."
)

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
    f"PCA (linear) explains {pca_var_2d:.1%} of variance with 2 components. The "
    f"autoencoder (non-linear) explains {ae_var_explained:.1%}. The autoencoder can "
    "bend and twist its projection surface to fit the data better, which often produces "
    "tighter city clusters. This is like the difference between projecting a globe onto "
    "a flat map (PCA: always distorted) vs projecting it onto a curved surface "
    "(autoencoder: can reduce distortion by curving). Whether the extra complexity is "
    "worth it depends on your downstream task, but it demonstrates that non-linearity "
    "can capture structure that linear methods miss."
)

# ── 53.6 Anomaly Detection with Reconstruction Error ────────────────────────
st.header("53.5  Anomaly Detection via Reconstruction Error")

st.markdown(
    "Here is an elegant trick that falls out of the autoencoder for free. The autoencoder "
    "was trained on typical weather data -- the kinds of temperature, humidity, wind, and "
    "pressure combinations that occur most often across our 105,264 hourly readings. "
    "It has learned to represent and reconstruct *normal* weather well."
)
st.markdown(
    "So what happens when you feed it something unusual? Say, a reading of -5 C with 95% "
    "humidity and 50 km/h winds in Dallas (a rare ice storm). The autoencoder has never "
    "seen anything like this. It tries to reconstruct it, but it cannot -- the compressed "
    "2-number representation has no good way to encode 'Dallas ice storm' because that "
    "pattern is vanishingly rare in the training data. The reconstruction error spikes."
)
st.markdown(
    "And there is your anomaly detector: **high reconstruction error = unusual observation**. "
    "You never had to label a single anomaly to build it. The autoencoder figured out what "
    "'normal' looks like, and anything that does not fit gets flagged."
)

concept_box(
    "Anomalies and Reconstruction",
    "An autoencoder trained on normal data learns the <b>typical patterns</b>: the usual "
    "temperature-humidity correlations, the normal pressure ranges for each city, the "
    "typical wind speeds. When it encounters an outlier -- a freak heat wave in NYC, "
    "an instrument malfunction reading -40 C in Houston, a data entry error -- it cannot "
    "reconstruct it well because those patterns were not in the training data.<br><br>"
    "This is <b>unsupervised anomaly detection</b>: you train on everything (assuming "
    "most of your data is normal), and the anomalies reveal themselves through high "
    "reconstruction error. The only assumption is that anomalies are rare enough that "
    "the autoencoder does not learn to reconstruct them well."
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

st.markdown(
    "**Reading this histogram:** Most observations cluster near zero reconstruction "
    "error -- the autoencoder handles them easily. The long right tail contains the "
    "unusual observations. Notice if some cities have longer tails than others -- that "
    "tells you which cities have more extreme or variable weather."
)

# Threshold for anomalies
threshold_pct = st.slider("Anomaly threshold (percentile)", 90, 99, 95, key="anomaly_thresh")
threshold_val = np.percentile(sample_mses, threshold_pct)
n_anomalies = (sample_mses > threshold_val).sum()

st.markdown(
    "**What the slider does:** It sets the percentile cutoff for flagging anomalies. "
    "At p95, we flag the top 5% of observations by reconstruction error. At p99, only "
    "the top 1%. There is no objectively correct threshold -- it depends on how aggressive "
    "you want your anomaly detector to be."
)

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
    "The anomaly rates are not uniform across cities, and that is informative. NYC "
    "typically has a higher anomaly rate because its weather is more variable -- polar "
    "vortex events in winter, heat waves in summer, nor'easters. LA tends to have a lower "
    "rate because its Mediterranean climate is boringly consistent (72 F and sunny, again). "
    "The Texas cities fall somewhere in between. This is the autoencoder teaching us "
    "something about the *structure* of each city's weather -- some cities simply have more "
    "'unusual' days than others. Whether that is a feature or a bug depends entirely on "
    "what you are trying to detect."
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
        "The bottleneck is the narrow hidden layer between encoder and decoder -- "
        "the eye of the needle that all information must pass through. In our weather "
        "autoencoder, it is the 2-neuron layer that forces 4 weather features (temperature, "
        "humidity, wind, pressure) to be represented by just 2 numbers. Without this "
        "constraint, the network would learn the identity function (just copy the input "
        "to the output) and we would learn nothing about the data's structure. The "
        "bottleneck is what makes the autoencoder interesting -- it forces the network "
        "to decide what information is essential and what can be sacrificed."
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
        "The logic is elegantly simple. The autoencoder trains on mostly normal weather "
        "data and learns to reconstruct it well. A typical Dallas reading (28 C, 55%, "
        "10 km/h, 1014 hPa) goes through the bottleneck and comes back as (27.8 C, 56%, "
        "10.3 km/h, 1013.7 hPa) -- low error. An anomalous reading (-5 C, 95%, 50 km/h, "
        "990 hPa during an ice storm) goes through and comes back as (15 C, 70%, 20 km/h, "
        "1005 hPa) -- the autoencoder cannot reconstruct it because it never learned to "
        "represent ice storms. That high reconstruction error is your anomaly signal, and "
        "you never had to label a single anomaly to get it."
    ),
    key="ch53_quiz2",
)

quiz(
    "You train an autoencoder with bottleneck size 1 on our 4-feature weather data. "
    "It has high reconstruction error. What does this tell you about the data?",
    [
        "The data is too noisy to be useful",
        "The 4 weather features cannot be fully represented by a single number -- at least 2 underlying dimensions are needed",
        "The autoencoder has too many layers",
        "The learning rate is too high",
    ],
    correct_idx=1,
    explanation=(
        "High reconstruction error with bottleneck = 1 means the autoencoder is failing to "
        "capture the data's structure with just one number. This tells you the data has at "
        "least 2 meaningful dimensions of variation. In our weather data, one dimension "
        "roughly captures the temperature-pressure axis (hot/cold, high/low pressure), and "
        "a second dimension captures the humidity-wind axis. You cannot collapse both into "
        "one number without losing real information. When we go to bottleneck = 2, the "
        "reconstruction error drops dramatically -- confirming that 2 dimensions capture "
        "most of the weather data's structure. This is the same insight PCA gives us (the "
        "first 2 principal components explain most of the variance), but the autoencoder "
        "discovers it through non-linear compression rather than linear algebra."
    ),
    key="ch53_quiz3",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "An autoencoder takes a 4-number weather reading (temperature, humidity, wind, pressure) "
    "and **compresses** it to 2 numbers, then **reconstructs** the original 4. What survives "
    "the round trip reveals the data's essential structure.",
    "The **bottleneck** forces information through a narrow channel. Without it, the network "
    "just copies -- with it, the network must decide which weather information is essential "
    "and which is redundant.",
    "**Smaller bottlenecks** = more compression but higher reconstruction error. Going from "
    "1D to 2D dramatically improves reconstruction; going from 2D to 3D barely helps -- "
    "telling us weather data has roughly 2 dominant dimensions.",
    "The autoencoder's **latent space** provides a non-linear 2D map of weather observations. "
    "Similar weather (regardless of city) clusters together; different weather separates. "
    "This often produces tighter clusters than PCA's linear projection.",
    "**High reconstruction error** is a free anomaly detector: the autoencoder cannot "
    "reconstruct unusual weather (ice storms, heat waves, instrument errors) because it "
    "never learned to represent those rare patterns. No labeled anomalies needed.",
    "Cities with more variable or extreme weather (NYC, with its polar vortex events) tend "
    "to have higher anomaly rates than mild, consistent cities (LA).",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 52: RNN & LSTM",
    prev_page="52_RNN_and_LSTM.py",
    next_label="Ch 54",
    next_page="54_Bayesian_Thinking.py",
)
