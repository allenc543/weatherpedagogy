"""Chapter 29: DBSCAN -- Density-based clustering, noise detection, and irregular shapes."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(29, "DBSCAN", part="VI")
st.markdown(
    "DBSCAN finds clusters based on **density** rather than distance to centroids. "
    "It can discover arbitrarily shaped clusters and automatically labels **noise "
    "points** -- perfect for detecting extreme weather events."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
filt = sidebar_filters(df)

# Subsample for speed
sample_size = min(8000, len(filt))
sample = filt.sample(sample_size, random_state=42).copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(sample[FEATURE_COLS])

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 -- How DBSCAN Works
# ══════════════════════════════════════════════════════════════════════════════
st.header("1. How DBSCAN Works")

concept_box(
    "Density-Based Spatial Clustering",
    "DBSCAN defines clusters as dense regions separated by sparse regions. Two key parameters:<br>"
    "- <b>epsilon (eps)</b>: The radius of the neighborhood around each point<br>"
    "- <b>min_samples</b>: Minimum points within epsilon to be a 'core' point<br><br>"
    "Point types:<br>"
    "- <b>Core point</b>: Has >= min_samples neighbors within eps<br>"
    "- <b>Border point</b>: Within eps of a core point, but not core itself<br>"
    "- <b>Noise point</b>: Neither core nor border -- an outlier!"
)

st.markdown("""
**The Algorithm:**
1. For each point, count neighbors within distance epsilon
2. Points with >= min_samples neighbors are **core points**
3. Connect core points that are within epsilon of each other into clusters
4. Assign border points to the nearest core point's cluster
5. Label remaining points as **noise** (label = -1)
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 -- Choosing Epsilon: K-Distance Plot
# ══════════════════════════════════════════════════════════════════════════════
st.header("2. Choosing Epsilon: K-Distance Plot")

concept_box(
    "The K-Distance Method",
    "To find a good epsilon, compute the distance to the K-th nearest neighbor for "
    "every point, sort these distances, and look for the 'elbow'. The elbow value "
    "is a good starting point for epsilon."
)

k_dist = st.slider("K for distance calculation", 3, 20, 5, 1, key="db_kdist")

nn = NearestNeighbors(n_neighbors=k_dist)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])

fig_kdist = go.Figure()
fig_kdist.add_trace(go.Scatter(
    x=list(range(len(k_distances))), y=k_distances,
    mode="lines", line=dict(color="#2E86C1", width=2),
    name=f"{k_dist}-distance"
))
apply_common_layout(fig_kdist, title=f"K-Distance Plot (K={k_dist})", height=400)
fig_kdist.update_layout(
    xaxis_title="Points (sorted by distance)",
    yaxis_title=f"Distance to {k_dist}th Nearest Neighbor"
)
st.plotly_chart(fig_kdist, use_container_width=True)

# Suggest epsilon from the elbow
p90 = np.percentile(k_distances, 90)
st.markdown(
    f"The 90th percentile distance is **{p90:.2f}**. Points beyond this are in "
    "sparse regions and may be noise."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 -- Interactive DBSCAN
# ══════════════════════════════════════════════════════════════════════════════
st.header("3. Interactive DBSCAN Clustering")

col_p1, col_p2 = st.columns(2)
with col_p1:
    eps_val = st.slider("Epsilon (eps)", 0.1, 3.0, 0.8, 0.05, key="db_eps")
with col_p2:
    min_samp = st.slider("Minimum samples (min_samples)", 3, 50, 10, 1, key="db_minsamp")

dbscan = DBSCAN(eps=eps_val, min_samples=min_samp)
db_labels = dbscan.fit_predict(X_scaled)
sample["db_cluster"] = db_labels

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = (db_labels == -1).sum()
noise_pct = n_noise / len(db_labels) * 100

# Metrics (excluding noise)
mask_not_noise = db_labels != -1
if n_clusters > 1 and mask_not_noise.sum() > 0:
    ari = adjusted_rand_score(sample.loc[mask_not_noise.values, "city"] if hasattr(mask_not_noise, 'values') else sample["city"][mask_not_noise], db_labels[mask_not_noise])
    nmi = normalized_mutual_info_score(sample["city"][mask_not_noise], db_labels[mask_not_noise])
else:
    ari = 0.0
    nmi = 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Clusters Found", n_clusters)
col2.metric("Noise Points", f"{n_noise} ({noise_pct:.1f}%)")
col3.metric("ARI (non-noise)", f"{ari:.3f}")
col4.metric("NMI (non-noise)", f"{nmi:.3f}")

# Visualize
feat_x = st.selectbox("X-axis", FEATURE_COLS, index=0, key="db_x")
feat_y = st.selectbox("Y-axis", FEATURE_COLS, index=3, key="db_y")

# Color noise separately
sample["db_label_str"] = sample["db_cluster"].apply(
    lambda x: "Noise" if x == -1 else f"Cluster {x}"
)

col_v1, col_v2 = st.columns(2)
with col_v1:
    # Separate noise and cluster points for coloring
    noise_data = sample[sample["db_cluster"] == -1]
    cluster_data = sample[sample["db_cluster"] != -1]

    fig_db = go.Figure()
    # Plot cluster points
    if len(cluster_data) > 0:
        unique_clusters = sorted(cluster_data["db_cluster"].unique())
        colors = px.colors.qualitative.Set2
        for i, c in enumerate(unique_clusters):
            c_data = cluster_data[cluster_data["db_cluster"] == c]
            fig_db.add_trace(go.Scatter(
                x=c_data[feat_x], y=c_data[feat_y],
                mode="markers", name=f"Cluster {c}",
                marker=dict(color=colors[i % len(colors)], size=4, opacity=0.5)
            ))
    # Plot noise points
    if len(noise_data) > 0:
        fig_db.add_trace(go.Scatter(
            x=noise_data[feat_x], y=noise_data[feat_y],
            mode="markers", name="Noise",
            marker=dict(color="black", size=3, opacity=0.3, symbol="x")
        ))
    apply_common_layout(fig_db, title="DBSCAN Clusters", height=450)
    fig_db.update_layout(
        xaxis_title=FEATURE_LABELS.get(feat_x, feat_x),
        yaxis_title=FEATURE_LABELS.get(feat_y, feat_y)
    )
    st.plotly_chart(fig_db, use_container_width=True)

with col_v2:
    fig_actual = px.scatter(
        sample, x=feat_x, y=feat_y, color="city",
        color_discrete_map=CITY_COLORS,
        title="Actual City Labels",
        labels=FEATURE_LABELS, opacity=0.4,
    )
    apply_common_layout(fig_actual, title="Actual City Labels", height=450)
    st.plotly_chart(fig_actual, use_container_width=True)

if n_clusters == 0:
    warning_box(
        "No clusters found! Epsilon may be too small or min_samples too large. "
        "Try increasing epsilon or decreasing min_samples."
    )
elif noise_pct > 50:
    warning_box(
        f"Over {noise_pct:.0f}% of points are labeled as noise. Epsilon may be "
        "too small. Try increasing it."
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 -- Noise Points as Extreme Weather
# ══════════════════════════════════════════════════════════════════════════════
st.header("4. Noise Points as Extreme Weather Events")

concept_box(
    "DBSCAN's Built-in Anomaly Detection",
    "Points that DBSCAN labels as noise are in low-density regions -- they are "
    "unusual observations. In weather data, these might be extreme temperature days, "
    "unusual humidity, or storm events."
)

if n_noise > 0:
    noise_df = sample[sample["db_cluster"] == -1]
    normal_df = sample[sample["db_cluster"] != -1]

    st.subheader("Noise vs Normal Points: Feature Comparison")

    compare_stats = []
    for feat in FEATURE_COLS:
        compare_stats.append({
            "Feature": FEATURE_LABELS.get(feat, feat),
            "Normal Mean": normal_df[feat].mean() if len(normal_df) > 0 else np.nan,
            "Normal Std": normal_df[feat].std() if len(normal_df) > 0 else np.nan,
            "Noise Mean": noise_df[feat].mean(),
            "Noise Std": noise_df[feat].std(),
        })

    compare_df = pd.DataFrame(compare_stats).round(2)
    st.dataframe(compare_df, use_container_width=True)

    # Distribution of noise points by city
    noise_by_city = noise_df["city"].value_counts()
    fig_noise_city = px.bar(
        x=noise_by_city.index, y=noise_by_city.values,
        title="Noise Points by City",
        labels={"x": "City", "y": "Noise Count"},
        color=noise_by_city.index.tolist(),
        color_discrete_map=CITY_COLORS
    )
    apply_common_layout(fig_noise_city, title="Noise Points by City", height=350)
    fig_noise_city.update_layout(showlegend=False)
    st.plotly_chart(fig_noise_city, use_container_width=True)

    insight_box(
        "Noise points often represent extreme weather: unusually hot days, storms, or "
        "temperature anomalies. DBSCAN discovers these automatically!"
    )
else:
    st.info("No noise points detected with current parameters. Try decreasing epsilon or increasing min_samples.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 -- Parameter Sensitivity
# ══════════════════════════════════════════════════════════════════════════════
st.header("5. Parameter Sensitivity")

st.markdown("See how the number of clusters and noise fraction change with epsilon.")

eps_range = np.arange(0.2, 2.5, 0.1)
param_results = []
for e in eps_range:
    db_temp = DBSCAN(eps=e, min_samples=min_samp)
    labels_temp = db_temp.fit_predict(X_scaled)
    n_c = len(set(labels_temp)) - (1 if -1 in labels_temp else 0)
    n_n = (labels_temp == -1).sum()
    param_results.append({
        "Epsilon": round(e, 2),
        "Clusters": n_c,
        "Noise %": n_n / len(labels_temp) * 100,
    })

param_df = pd.DataFrame(param_results)

fig_param = go.Figure()
fig_param.add_trace(go.Scatter(
    x=param_df["Epsilon"], y=param_df["Clusters"],
    mode="lines+markers", name="Clusters",
    line=dict(color="#2E86C1", width=3), yaxis="y1"
))
fig_param.add_trace(go.Scatter(
    x=param_df["Epsilon"], y=param_df["Noise %"],
    mode="lines+markers", name="Noise %",
    line=dict(color="#E63946", width=3), yaxis="y2"
))
fig_param.update_layout(
    title=f"DBSCAN Sensitivity to Epsilon (min_samples={min_samp})",
    xaxis=dict(title="Epsilon"),
    yaxis=dict(title="Number of Clusters", side="left", titlefont=dict(color="#2E86C1")),
    yaxis2=dict(title="Noise %", side="right", overlaying="y", titlefont=dict(color="#E63946")),
    template="plotly_white", height=400,
)
st.plotly_chart(fig_param, use_container_width=True)

insight_box(
    "Small epsilon = many small clusters + lots of noise. Large epsilon = fewer "
    "clusters as everything merges. The right balance depends on the data's density structure."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 -- DBSCAN vs K-Means
# ══════════════════════════════════════════════════════════════════════════════
st.header("6. DBSCAN vs K-Means")

st.markdown("""
| Aspect | K-Means | DBSCAN |
|--------|---------|--------|
| Requires K? | Yes | No (finds clusters automatically) |
| Cluster shape | Spherical | Arbitrary |
| Handles noise? | No (assigns everything) | Yes (labels outliers) |
| Parameters | K | eps, min_samples |
| Sensitivity | To initialization | To eps and min_samples |
| Equal-size clusters? | Tends to create them | No requirement |
""")

# Compare quantitatively
km_6 = KMeans(n_clusters=6, random_state=42, n_init=10)
km_labels = km_6.fit_predict(X_scaled)
km_ari = adjusted_rand_score(sample["city"], km_labels)

comp_data = pd.DataFrame({
    "Algorithm": ["K-Means (K=6)", f"DBSCAN (eps={eps_val}, ms={min_samp})"],
    "Clusters": [6, n_clusters],
    "Noise Points": [0, n_noise],
    "ARI": [km_ari, ari],
})
st.dataframe(comp_data, use_container_width=True)

warning_box(
    "DBSCAN struggles when clusters have very different densities. The Texas cities "
    "and NYC may have different weather variability, requiring different epsilon values."
)

code_example("""
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-distance plot to choose epsilon
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])
# Plot k_distances and look for the elbow

# Fit DBSCAN
db = DBSCAN(eps=0.8, min_samples=10)
labels = db.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_mask = labels == -1
print(f"Clusters: {n_clusters}, Noise: {noise_mask.sum()}")

# Noise points = potential anomalies!
anomalies = X[noise_mask]
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 -- Quiz & Takeaways
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

quiz(
    "What is unique about DBSCAN compared to K-Means?",
    [
        "It always finds exactly 6 clusters",
        "It can identify noise points (outliers) and does not require K to be specified upfront",
        "It is always faster",
        "It does not require feature scaling",
    ],
    correct_idx=1,
    explanation="DBSCAN discovers clusters based on density, automatically determines the number of clusters, and labels low-density points as noise.",
    key="q_db_1"
)

quiz(
    "What happens when you decrease epsilon in DBSCAN?",
    [
        "Fewer, larger clusters",
        "More clusters and more noise points (points in sparse areas become noise)",
        "No effect on the results",
        "The algorithm runs faster",
    ],
    correct_idx=1,
    explanation="Smaller epsilon means each point needs closer neighbors to form a cluster. Points in sparser regions get labeled as noise.",
    key="q_db_2"
)

takeaways([
    "DBSCAN finds clusters based on point density, not distance to centroids.",
    "It automatically determines the number of clusters and labels outliers as noise.",
    "Two key parameters: epsilon (neighborhood radius) and min_samples (density threshold).",
    "The K-distance plot helps choose a good epsilon value.",
    "Noise points in weather data often correspond to extreme weather events.",
    "DBSCAN struggles with clusters of varying density, unlike K-Means which assumes equal-size clusters.",
])
