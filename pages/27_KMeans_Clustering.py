"""Chapter 27: K-Means Clustering -- Centroids, the elbow method, and unsupervised discovery."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(27, "K-Means Clustering", part="VI")
st.markdown(
    "Everything we have done so far has been supervised: we had city labels and trained "
    "models to predict them. Now comes the interesting question: what if you ripped off "
    "all the labels? Could an algorithm discover that there are different types of "
    "weather just by looking at the numbers? K-Means says yes, and it does it with an "
    "almost embarrassingly simple algorithm: pick K random centers, assign each point "
    "to its nearest center, move the centers to the middle of their assigned points, "
    "repeat. That is it. And yet it discovers **real structure** in the data."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
filt = sidebar_filters(df)

# Subsample for speed
sample_size = min(10000, len(filt))
sample = filt.sample(sample_size, random_state=42)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(sample[FEATURE_COLS])
X_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS, index=sample.index)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 -- How K-Means Works
# ══════════════════════════════════════════════════════════════════════════════
st.header("1. How K-Means Works")

concept_box(
    "The K-Means Algorithm",
    "The algorithm is beautifully simple:<br>"
    "1. <b>Initialize</b> K centroids (randomly, or using a smarter scheme like k-means++)<br>"
    "2. <b>Assign</b> each point to the nearest centroid<br>"
    "3. <b>Update</b> each centroid to the mean of its assigned points<br>"
    "4. <b>Repeat</b> steps 2-3 until convergence (assignments stop changing)<br><br>"
    "That is the whole algorithm. Each iteration, the centroids slide toward the 'center "
    "of gravity' of their cluster, like balls rolling downhill. Convergence is guaranteed, "
    "usually in fewer than 10 iterations."
)

formula_box(
    "Objective: Minimize Within-Cluster Sum of Squares (Inertia)",
    r"\min \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2",
    "Where C_k is cluster k and mu_k is its centroid. K-Means minimizes the total "
    "squared distance from points to their assigned centroids. It is a greedy algorithm -- "
    "it finds a local minimum, not necessarily the global one, which is why running it "
    "multiple times with different initializations is standard practice."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 -- Interactive K-Means
# ══════════════════════════════════════════════════════════════════════════════
st.header("2. Interactive K-Means Clustering")

K = st.slider("Number of clusters (K)", 2, 10, 6, 1, key="km_k")

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
sample["cluster"] = cluster_labels.astype(str)

# Metrics
inertia = kmeans.inertia_
ari = adjusted_rand_score(sample["city"], cluster_labels)
nmi = normalized_mutual_info_score(sample["city"], cluster_labels)

col1, col2, col3 = st.columns(3)
col1.metric("Inertia", f"{inertia:,.0f}")
col2.metric("Adjusted Rand Index", f"{ari:.3f}")
col3.metric("Normalized Mutual Info", f"{nmi:.3f}")

st.caption(
    "ARI and NMI compare cluster assignments to actual city labels (which K-Means "
    "never saw). 1.0 = the clusters perfectly match the cities. 0.0 = random noise. "
    "We are cheating a little by using labels to evaluate an unsupervised algorithm, "
    "but it is the best way to see how well the algorithm is doing."
)

# Scatter plot of clusters
feat_x = st.selectbox("X-axis", FEATURE_COLS, index=0, key="km_x")
feat_y = st.selectbox("Y-axis", FEATURE_COLS, index=3, key="km_y")

col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    fig_cluster = px.scatter(
        sample, x=feat_x, y=feat_y, color="cluster",
        title="K-Means Clusters",
        labels=FEATURE_LABELS, opacity=0.4,
        category_orders={"cluster": [str(i) for i in range(K)]}
    )
    # Add centroids (unscaled)
    centroids_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
    feat_x_idx = FEATURE_COLS.index(feat_x)
    feat_y_idx = FEATURE_COLS.index(feat_y)
    fig_cluster.add_trace(go.Scatter(
        x=centroids_unscaled[:, feat_x_idx],
        y=centroids_unscaled[:, feat_y_idx],
        mode="markers", name="Centroids",
        marker=dict(color="black", size=15, symbol="x", line=dict(width=2))
    ))
    apply_common_layout(fig_cluster, title="K-Means Clusters", height=450)
    st.plotly_chart(fig_cluster, use_container_width=True)

with col_viz2:
    fig_actual = px.scatter(
        sample, x=feat_x, y=feat_y, color="city",
        color_discrete_map=CITY_COLORS,
        title="Actual City Labels",
        labels=FEATURE_LABELS, opacity=0.4,
    )
    apply_common_layout(fig_actual, title="Actual City Labels", height=450)
    st.plotly_chart(fig_actual, use_container_width=True)

insight_box(
    f"With K={K}, the Adjusted Rand Index is **{ari:.3f}**. The algorithm has never "
    "seen a single city label, yet it is discovering weather patterns that roughly "
    "correspond to geographic reality. It cannot tell you 'this is Houston' -- it "
    "does not know what Houston is -- but it can tell you 'these observations look "
    "similar to each other and different from those observations.' That is unsupervised "
    "learning in a nutshell."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 -- Elbow Method
# ══════════════════════════════════════════════════════════════════════════════
st.header("3. The Elbow Method: Choosing K")

concept_box(
    "How to Choose the Number of Clusters",
    "You might object: we have to specify K upfront, but how do we know the right K? "
    "Fair question. The <b>elbow method</b> plots inertia (total within-cluster distance) "
    "against K. As K increases, inertia always decreases (more clusters = shorter "
    "distances to centroids). But at some point, the decrease slows dramatically -- "
    "you are splitting natural clusters that do not want to be split. That inflection "
    "point is the 'elbow,' and it suggests the natural number of clusters."
)

k_range = range(2, 11)
inertias = []
aris = []
for k in k_range:
    km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_temp.fit(X_scaled)
    inertias.append(km_temp.inertia_)
    aris.append(adjusted_rand_score(sample["city"], km_temp.labels_))

elbow_df = pd.DataFrame({
    "K": list(k_range),
    "Inertia": inertias,
    "Adjusted Rand Index": aris,
})

fig_elbow = go.Figure()
fig_elbow.add_trace(go.Scatter(
    x=elbow_df["K"], y=elbow_df["Inertia"],
    mode="lines+markers", name="Inertia",
    line=dict(color="#2E86C1", width=3), marker=dict(size=10)
))
fig_elbow.add_vline(x=6, line_dash="dash", line_color="gray",
                     annotation_text="K=6 (actual cities)")
apply_common_layout(fig_elbow, title="Elbow Plot: Inertia vs K", height=400)
fig_elbow.update_layout(xaxis_title="Number of Clusters (K)", yaxis_title="Inertia")
st.plotly_chart(fig_elbow, use_container_width=True)

# ARI vs K
fig_ari = px.line(
    elbow_df, x="K", y="Adjusted Rand Index",
    title="Cluster Quality (ARI) vs K",
    markers=True,
)
fig_ari.update_traces(line=dict(color="#E63946", width=3))
fig_ari.add_vline(x=6, line_dash="dash", line_color="gray",
                   annotation_text="K=6 (actual cities)")
apply_common_layout(fig_ari, title="Cluster Quality (ARI) vs K", height=400)
st.plotly_chart(fig_ari, use_container_width=True)

insight_box(
    "The elbow is often around K=4-6, which is encouraging -- with 6 actual cities, "
    "K=6 is a natural choice. But here is the subtle point: the 4 Texas cities share "
    "such similar weather that the algorithm sometimes prefers K=3 or K=4, lumping Texas "
    "into one or two clusters. From a pure weather-pattern perspective, that might "
    "actually be the 'right' answer. The clustering is not wrong -- our label granularity "
    "is finer than the natural structure of the data."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 -- Cluster Centroids
# ══════════════════════════════════════════════════════════════════════════════
st.header("4. Understanding Cluster Centroids")

st.markdown(
    "Each centroid is the average weather profile of its cluster -- a 'prototype' "
    "observation that summarizes everything in that group. Let us compare them to "
    "the actual city averages and see how close the blind algorithm gets."
)

# K=6 centroids
kmeans_6 = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans_6.fit(X_scaled)
centroids_6 = scaler.inverse_transform(kmeans_6.cluster_centers_)

centroid_df = pd.DataFrame(
    centroids_6,
    columns=[FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
    index=[f"Cluster {i}" for i in range(6)]
)

st.subheader("K=6 Cluster Centroids (original scale)")
st.dataframe(centroid_df.round(2), use_container_width=True)

# Actual city means
city_means = filt.groupby("city")[FEATURE_COLS].mean()
city_means.columns = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS]

st.subheader("Actual City Means")
st.dataframe(city_means.round(2), use_container_width=True)

# Heatmap comparison
fig_hm = go.Figure()
fig_hm.add_trace(go.Heatmap(
    z=centroid_df.values,
    x=centroid_df.columns.tolist(),
    y=centroid_df.index.tolist(),
    colorscale="RdYlBu_r",
    text=centroid_df.values.round(1),
    texttemplate="%{text}",
))
apply_common_layout(fig_hm, title="Cluster Centroids Heatmap", height=400)
st.plotly_chart(fig_hm, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 -- Cluster-to-City Mapping
# ══════════════════════════════════════════════════════════════════════════════
st.header("5. Comparing K=6 Clusters to Actual Cities")

# Cross-tabulation
labels_6 = kmeans_6.labels_
cross_tab = pd.crosstab(sample["city"], labels_6, colnames=["Cluster"])
st.markdown("**Cross-tabulation: how cities distribute across clusters**")
st.dataframe(cross_tab, use_container_width=True)

# Normalized cross-tab (percentage)
cross_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0).round(3) * 100
fig_cross = px.imshow(
    cross_pct.values,
    x=[f"Cluster {i}" for i in range(6)],
    y=cross_pct.index.tolist(),
    color_continuous_scale="Blues",
    title="City Distribution Across Clusters (%)",
    text_auto=".1f", aspect="auto",
    labels=dict(color="%")
)
apply_common_layout(fig_cross, title="City Distribution Across Clusters (%)", height=400)
st.plotly_chart(fig_cross, use_container_width=True)

warning_box(
    "K-Means assumes clusters are spherical (round blobs) and roughly equal in size. "
    "Real weather data does not care about your assumptions. The Texas cities form an "
    "amorphous, overlapping blob that K-Means cheerfully slices into spherical chunks, "
    "which is why some Texas cities get merged together or split in odd ways."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 -- Iteration Visualization
# ══════════════════════════════════════════════════════════════════════════════
st.header("6. K-Means Iteration by Iteration")

st.markdown(
    "Let us watch the algorithm think. Below, you can see how centroids start at random "
    "positions and gradually slide toward the centers of their clusters. Most of the "
    "action happens in the first 2-3 iterations; after that, the centroids are barely "
    "moving."
)

n_iter_show = st.slider("Number of iterations to show", 1, 10, 5, 1, key="km_iter")

# Run K-Means step by step
np.random.seed(42)
init_centroids = X_scaled[np.random.choice(len(X_scaled), K, replace=False)]

centroid_history = [init_centroids.copy()]
current_centroids = init_centroids.copy()

for iteration in range(n_iter_show):
    # Assign
    distances = np.linalg.norm(X_scaled[:, np.newaxis] - current_centroids[np.newaxis, :], axis=2)
    assignments = distances.argmin(axis=1)
    # Update
    new_centroids = np.array([
        X_scaled[assignments == k].mean(axis=0) if (assignments == k).sum() > 0 else current_centroids[k]
        for k in range(K)
    ])
    current_centroids = new_centroids
    centroid_history.append(current_centroids.copy())

# Show the centroid movement
fig_iter = go.Figure()

# Plot final assignments
final_assignments = np.linalg.norm(
    X_scaled[:, np.newaxis] - current_centroids[np.newaxis, :], axis=2
).argmin(axis=1)

feat_x_idx = FEATURE_COLS.index(feat_x)
feat_y_idx = FEATURE_COLS.index(feat_y)

# Plot points colored by final assignment
for k in range(K):
    mask = final_assignments == k
    fig_iter.add_trace(go.Scatter(
        x=X_scaled[mask, feat_x_idx], y=X_scaled[mask, feat_y_idx],
        mode="markers", name=f"Cluster {k}", opacity=0.2,
        marker=dict(size=3), showlegend=True
    ))

# Plot centroid trajectories
for k in range(K):
    traj_x = [centroid_history[i][k, feat_x_idx] for i in range(len(centroid_history))]
    traj_y = [centroid_history[i][k, feat_y_idx] for i in range(len(centroid_history))]
    fig_iter.add_trace(go.Scatter(
        x=traj_x, y=traj_y, mode="lines+markers",
        name=f"Centroid {k} path",
        line=dict(width=2, dash="dot"),
        marker=dict(size=8, symbol="diamond"),
        showlegend=False
    ))
    # Final centroid
    fig_iter.add_trace(go.Scatter(
        x=[traj_x[-1]], y=[traj_y[-1]], mode="markers",
        name=f"Final centroid {k}",
        marker=dict(size=15, symbol="x", line=dict(width=2), color="black"),
        showlegend=False
    ))

apply_common_layout(fig_iter, title=f"K-Means After {n_iter_show} Iterations (scaled features)", height=500)
fig_iter.update_layout(
    xaxis_title=f"{FEATURE_LABELS.get(feat_x, feat_x)} (scaled)",
    yaxis_title=f"{FEATURE_LABELS.get(feat_y, feat_y)} (scaled)"
)
st.plotly_chart(fig_iter, use_container_width=True)

insight_box(
    "The centroid trajectories tell the story: big jumps at first, then tiny adjustments. "
    "K-Means typically converges in fewer than 10 iterations. The diamond markers trace "
    "each centroid's path from its random starting position to its final resting place."
)

code_example("""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Scale features (important for distance-based methods)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Fit K-Means with chosen K
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Centroids in original scale
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 -- Quiz & Takeaways
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

quiz(
    "What does the 'elbow' in the elbow method represent?",
    [
        "The point where K-Means fails",
        "The point where adding more clusters gives diminishing returns in reducing inertia",
        "The optimal number of features",
        "The point where accuracy reaches 100%",
    ],
    correct_idx=1,
    explanation="Before the elbow, adding clusters splits real groups and gives big inertia drops. "
    "After the elbow, you are splitting groups that did not want to be split, and the returns "
    "are marginal. The elbow is where you stop getting meaningful structure.",
    key="q_km_1"
)

quiz(
    "K-Means is an unsupervised algorithm. What does that mean?",
    [
        "It does not use a computer",
        "It learns from data without labeled outcomes (no city labels needed)",
        "It does not use features",
        "It requires manual labeling of every point",
    ],
    correct_idx=1,
    explanation="Unsupervised means the algorithm discovers structure in data without anyone "
    "telling it the 'right answer.' It never sees city labels -- it just finds groups of "
    "similar weather observations and calls them clusters.",
    key="q_km_2"
)

takeaways([
    "K-Means groups data into K clusters by minimizing within-cluster distances. The algorithm is embarrassingly simple and remarkably effective.",
    "The algorithm alternates between assigning points to centroids and moving centroids to cluster centers. Convergence is guaranteed and usually fast.",
    "The elbow method helps choose K: plot inertia vs K and look for the diminishing-returns inflection point.",
    "K=6 clusters roughly correspond to our 6 cities, but the 4 Texas cities often merge -- the algorithm is telling us they genuinely have similar weather.",
    "Feature scaling is essential: K-Means uses Euclidean distance, so unscaled features cause the same problems as KNN.",
    "K-Means assumes spherical, equal-size clusters. Real data is rarely this cooperative, which is why DBSCAN and hierarchical clustering exist.",
])
