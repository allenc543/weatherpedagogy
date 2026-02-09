"""Chapter 28: Hierarchical Clustering -- Dendrograms and linkage methods."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS, TEXAS_CITIES
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(28, "Hierarchical Clustering", part="VI")
st.markdown(
    "Hierarchical clustering builds a **tree of clusters** (dendrogram) that "
    "reveals the nested structure in data. Unlike K-Means, you do not need to "
    "choose K upfront."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
filt = sidebar_filters(df)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 -- How Hierarchical Clustering Works
# ══════════════════════════════════════════════════════════════════════════════
st.header("1. How Hierarchical Clustering Works")

concept_box(
    "Agglomerative (Bottom-Up) Approach",
    "1. Start with each data point as its own cluster<br>"
    "2. Find the two closest clusters and merge them<br>"
    "3. Repeat until all points are in a single cluster<br>"
    "4. Cut the dendrogram at the desired level to get K clusters<br><br>"
    "The key question: how do you define 'closest' between clusters?"
)

st.markdown("""
**Linkage methods** determine how distance between clusters is measured:
- **Single linkage**: Minimum distance between any two points in different clusters (can create elongated chains)
- **Complete linkage**: Maximum distance between any two points (creates compact clusters)
- **Average linkage**: Average distance between all pairs of points
- **Ward's method**: Minimizes the increase in total within-cluster variance (similar to K-Means objective)
""")

formula_box(
    "Ward's Linkage",
    r"d(C_i, C_j) = \sqrt{\frac{2 n_i n_j}{n_i + n_j}} \|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|",
    "Merges the pair of clusters that causes the smallest increase in total within-cluster variance."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 -- Dendrogram of City Averages
# ══════════════════════════════════════════════════════════════════════════════
st.header("2. Dendrogram: How Cities Cluster")

st.markdown(
    "We compute the mean weather profile for each city and build a dendrogram "
    "showing which cities are most similar."
)

linkage_method = st.selectbox(
    "Linkage method",
    ["ward", "complete", "average", "single"],
    index=0,
    key="hc_linkage"
)

# Compute city-level means
city_means = filt.groupby("city")[FEATURE_COLS].mean()
scaler_city = StandardScaler()
city_means_scaled = pd.DataFrame(
    scaler_city.fit_transform(city_means),
    index=city_means.index,
    columns=city_means.columns
)

# Scipy dendrogram
Z = linkage(city_means_scaled.values, method=linkage_method)

# Use plotly figure_factory for dendrogram
fig_dendro = ff.create_dendrogram(
    city_means_scaled.values,
    labels=city_means_scaled.index.tolist(),
    linkagefun=lambda x: linkage(x, method=linkage_method),
    color_threshold=0.7 * max(Z[:, 2]),
)
fig_dendro.update_layout(
    title=f"Dendrogram of City Weather Profiles ({linkage_method.title()} Linkage)",
    template="plotly_white",
    height=450,
    xaxis_title="City",
    yaxis_title="Distance",
)
st.plotly_chart(fig_dendro, use_container_width=True)

insight_box(
    "The Texas cities (Dallas, Houston, San Antonio, Austin) cluster together because "
    "they share similar subtropical weather. NYC and LA form their own branches "
    "because they have distinct climates."
)

# City profile heatmap
st.subheader("City Weather Profiles (Scaled)")
fig_prof = px.imshow(
    city_means_scaled.values,
    x=[FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
    y=city_means_scaled.index.tolist(),
    color_continuous_scale="RdBu_r",
    title="Standardized City Weather Profiles",
    text_auto=".2f", aspect="auto"
)
apply_common_layout(fig_prof, title="Standardized City Weather Profiles", height=350)
st.plotly_chart(fig_prof, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 -- Full Sample Clustering
# ══════════════════════════════════════════════════════════════════════════════
st.header("3. Clustering on Individual Observations")

st.markdown(
    "Now let's apply hierarchical clustering to individual weather observations "
    "and compare to actual city labels."
)

n_clusters = st.slider("Number of clusters", 2, 10, 6, 1, key="hc_n_clusters")

# Subsample for speed
sample_size = min(5000, len(filt))
sample = filt.sample(sample_size, random_state=42).copy()

scaler_samp = StandardScaler()
X_scaled = scaler_samp.fit_transform(sample[FEATURE_COLS])

hc_model = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage=linkage_method
)
hc_labels = hc_model.fit_predict(X_scaled)
sample["hc_cluster"] = hc_labels.astype(str)

ari = adjusted_rand_score(sample["city"], hc_labels)
nmi = normalized_mutual_info_score(sample["city"], hc_labels)

col1, col2, col3 = st.columns(3)
col1.metric("Number of Clusters", n_clusters)
col2.metric("Adjusted Rand Index", f"{ari:.3f}")
col3.metric("Normalized Mutual Info", f"{nmi:.3f}")

# Side-by-side comparison
feat_x = st.selectbox("X-axis", FEATURE_COLS, index=0, key="hc_x")
feat_y = st.selectbox("Y-axis", FEATURE_COLS, index=3, key="hc_y")

col_v1, col_v2 = st.columns(2)
with col_v1:
    fig_hc = px.scatter(
        sample, x=feat_x, y=feat_y, color="hc_cluster",
        title="Hierarchical Clusters",
        labels=FEATURE_LABELS, opacity=0.4,
        category_orders={"hc_cluster": [str(i) for i in range(n_clusters)]}
    )
    apply_common_layout(fig_hc, title="Hierarchical Clusters", height=400)
    st.plotly_chart(fig_hc, use_container_width=True)

with col_v2:
    fig_actual = px.scatter(
        sample, x=feat_x, y=feat_y, color="city",
        color_discrete_map=CITY_COLORS,
        title="Actual City Labels",
        labels=FEATURE_LABELS, opacity=0.4,
    )
    apply_common_layout(fig_actual, title="Actual City Labels", height=400)
    st.plotly_chart(fig_actual, use_container_width=True)

# Cross-tabulation
cross_tab = pd.crosstab(sample["city"], hc_labels, colnames=["Cluster"])
cross_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0).round(3) * 100

fig_cross = px.imshow(
    cross_pct.values,
    x=[f"Cluster {i}" for i in range(n_clusters)],
    y=cross_pct.index.tolist(),
    color_continuous_scale="Blues",
    title="City Distribution Across Hierarchical Clusters (%)",
    text_auto=".1f", aspect="auto",
    labels=dict(color="%")
)
apply_common_layout(fig_cross, title="City Distribution Across Hierarchical Clusters (%)", height=400)
st.plotly_chart(fig_cross, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 -- Linkage Method Comparison
# ══════════════════════════════════════════════════════════════════════════════
st.header("4. Comparing Linkage Methods")

st.markdown(
    "Different linkage methods can produce very different cluster assignments. "
    "Let's compare them all at K = 6 clusters."
)

linkage_methods = ["ward", "complete", "average", "single"]
linkage_results = []

for method in linkage_methods:
    hc_temp = AgglomerativeClustering(n_clusters=6, linkage=method)
    labels_temp = hc_temp.fit_predict(X_scaled)
    linkage_results.append({
        "Linkage": method.title(),
        "ARI": adjusted_rand_score(sample["city"], labels_temp),
        "NMI": normalized_mutual_info_score(sample["city"], labels_temp),
    })

linkage_df = pd.DataFrame(linkage_results)

fig_linkage = px.bar(
    linkage_df.melt(id_vars="Linkage", var_name="Metric", value_name="Score"),
    x="Linkage", y="Score", color="Metric", barmode="group",
    title="Cluster Quality by Linkage Method (K=6)",
    color_discrete_map={"ARI": "#2E86C1", "NMI": "#E63946"},
    text_auto=".3f"
)
apply_common_layout(fig_linkage, title="Cluster Quality by Linkage Method (K=6)", height=400)
st.plotly_chart(fig_linkage, use_container_width=True)

best_method = linkage_df.loc[linkage_df["ARI"].idxmax(), "Linkage"]
insight_box(
    f"**{best_method}** linkage produces the best agreement with actual city labels. "
    "Ward's method often performs well because it minimizes variance, similar to K-Means."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 -- Comparison to K-Means
# ══════════════════════════════════════════════════════════════════════════════
st.header("5. Hierarchical Clustering vs K-Means")

from sklearn.cluster import KMeans

km_compare = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
km_labels = km_compare.fit_predict(X_scaled)
km_ari = adjusted_rand_score(sample["city"], km_labels)
km_nmi = normalized_mutual_info_score(sample["city"], km_labels)

comparison = pd.DataFrame({
    "Method": ["K-Means", f"Hierarchical ({linkage_method})"],
    "ARI": [km_ari, ari],
    "NMI": [km_nmi, nmi],
})

fig_comp = px.bar(
    comparison.melt(id_vars="Method", var_name="Metric", value_name="Score"),
    x="Method", y="Score", color="Metric", barmode="group",
    title=f"K-Means vs Hierarchical Clustering (K={n_clusters})",
    color_discrete_map={"ARI": "#2E86C1", "NMI": "#E63946"},
    text_auto=".3f"
)
apply_common_layout(fig_comp, title=f"K-Means vs Hierarchical Clustering (K={n_clusters})", height=400)
st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("""
**Key differences:**

| Aspect | K-Means | Hierarchical |
|--------|---------|--------------|
| Needs K upfront? | Yes | No (cut dendrogram later) |
| Cluster shape | Spherical | Flexible |
| Scalability | Fast (large datasets) | Slow (O(n^2) memory) |
| Deterministic? | No (random init) | Yes |
| Produces hierarchy? | No | Yes (dendrogram) |
""")

warning_box(
    "Hierarchical clustering has O(n^2) memory and O(n^3) time complexity, making it "
    "impractical for very large datasets. That is why we subsample here."
)

code_example("""
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Agglomerative clustering
hc = AgglomerativeClustering(n_clusters=6, linkage='ward')
labels = hc.fit_predict(X_scaled)

# Dendrogram (on city averages for readability)
Z = linkage(city_means_scaled, method='ward')
dendrogram(Z, labels=city_names)
plt.title("City Dendrogram")
plt.show()

# Compare linkage methods
for method in ['ward', 'complete', 'average', 'single']:
    hc = AgglomerativeClustering(n_clusters=6, linkage=method)
    labels = hc.fit_predict(X_scaled)
    print(f"{method}: ARI = {adjusted_rand_score(y_true, labels):.3f}")
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 -- Quiz & Takeaways
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

quiz(
    "What is the main advantage of hierarchical clustering over K-Means?",
    [
        "It is always more accurate",
        "It produces a dendrogram that reveals nested cluster structure, and you don't need to choose K upfront",
        "It is faster on large datasets",
        "It requires no feature scaling",
    ],
    correct_idx=1,
    explanation="The dendrogram shows the full hierarchy of clusters at all levels, and you can choose K by cutting at different heights.",
    key="q_hc_1"
)

quiz(
    "Why do the Texas cities (Dallas, Houston, San Antonio, Austin) cluster together?",
    [
        "They have the same population",
        "They share similar subtropical weather patterns (temperature, humidity, pressure)",
        "They are all coastal cities",
        "They have the same elevation",
    ],
    correct_idx=1,
    explanation="Geographic proximity leads to similar weather. The Texas cities share a subtropical climate with similar temperature and humidity ranges.",
    key="q_hc_2"
)

takeaways([
    "Hierarchical clustering builds a tree of clusters (dendrogram) from bottom up.",
    "Linkage method determines how inter-cluster distance is measured (ward, complete, average, single).",
    "Dendrograms reveal nested structure: Texas cities form a tight sub-cluster.",
    "No need to choose K upfront -- cut the dendrogram at the desired level.",
    "Ward's linkage often works best and behaves similarly to K-Means.",
    "Main limitation: O(n^2) memory makes it impractical for very large datasets.",
])
