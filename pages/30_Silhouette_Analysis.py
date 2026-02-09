"""Chapter 30: Silhouette Analysis -- Measuring cluster quality and comparing algorithms."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(30, "Silhouette Analysis", part="VI")
st.markdown(
    "The silhouette score measures how well each point fits in its assigned cluster. "
    "It provides a rigorous way to **compare clustering algorithms** and choose the "
    "right number of clusters."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
filt = sidebar_filters(df)

# Subsample for speed
sample_size = min(5000, len(filt))
sample = filt.sample(sample_size, random_state=42).copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(sample[FEATURE_COLS])

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 -- What is the Silhouette Score?
# ══════════════════════════════════════════════════════════════════════════════
st.header("1. What is the Silhouette Score?")

concept_box(
    "Measuring Cluster Quality Without Labels",
    "The silhouette score evaluates clustering quality using only the data and "
    "cluster assignments -- no ground truth labels needed. For each point, it "
    "measures how similar it is to its own cluster vs. the nearest other cluster."
)

formula_box(
    "Silhouette Score for Point i",
    r"\underbrace{s(i)}_{\text{silhouette score}} = \frac{\underbrace{b(i)}_{\text{nearest cluster dist}} - \underbrace{a(i)}_{\text{own cluster dist}}}{\max(a(i),\; b(i))}",
    "a(i) = average distance to points in the SAME cluster (cohesion)\n"
    "b(i) = average distance to points in the NEAREST other cluster (separation)"
)

st.markdown("""
**Interpreting the silhouette score:**
- **s(i) close to +1**: The point is well-matched to its own cluster and poorly matched to neighboring clusters (good!)
- **s(i) close to 0**: The point is on or near the boundary between two clusters
- **s(i) close to -1**: The point is probably assigned to the wrong cluster

The **overall silhouette score** is the mean of all individual scores.
""")

# Simple illustration
st.subheader("Visual Intuition")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
    **High silhouette (s close to 1):**
    - Point is far from neighboring clusters
    - Point is tightly packed with its own cluster
    - a(i) is small, b(i) is large
    """)
with col_b:
    st.markdown("""
    **Low silhouette (s close to 0 or negative):**
    - Point is between clusters
    - Ambiguous assignment
    - a(i) is roughly equal to b(i)
    """)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 -- Silhouette Score vs K (K-Means)
# ══════════════════════════════════════════════════════════════════════════════
st.header("2. Silhouette Score vs K (K-Means)")

st.markdown(
    "Let's compute the silhouette score for different values of K using K-Means. "
    "The optimal K maximizes the silhouette score."
)

k_range = range(2, 11)
km_silhouettes = []
km_inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    km_silhouettes.append(sil)
    km_inertias.append(km.inertia_)

sil_df = pd.DataFrame({
    "K": list(k_range),
    "Silhouette Score": km_silhouettes,
    "Inertia": km_inertias,
})

# Dual axis: silhouette and inertia
fig_sil_k = go.Figure()
fig_sil_k.add_trace(go.Scatter(
    x=sil_df["K"], y=sil_df["Silhouette Score"],
    mode="lines+markers", name="Silhouette Score",
    line=dict(color="#2E86C1", width=3), marker=dict(size=10),
))
fig_sil_k.add_trace(go.Scatter(
    x=sil_df["K"], y=sil_df["Inertia"] / sil_df["Inertia"].max(),
    mode="lines+markers", name="Inertia (normalized)",
    line=dict(color="#E63946", width=3, dash="dash"), marker=dict(size=8),
    yaxis="y2"
))
fig_sil_k.update_layout(
    title="K-Means: Silhouette Score and Inertia vs K",
    xaxis=dict(title="Number of Clusters (K)", dtick=1),
    yaxis=dict(title="Silhouette Score", side="left", titlefont=dict(color="#2E86C1")),
    yaxis2=dict(title="Normalized Inertia", side="right", overlaying="y",
                titlefont=dict(color="#E63946")),
    template="plotly_white", height=450,
)
st.plotly_chart(fig_sil_k, use_container_width=True)

best_k_idx = np.argmax(km_silhouettes)
best_k = list(k_range)[best_k_idx]
best_sil = km_silhouettes[best_k_idx]

insight_box(
    f"The best silhouette score is **{best_sil:.3f}** at **K = {best_k}**. "
    "Note that the optimal K by silhouette may differ from the elbow method. "
    "Silhouette directly measures cluster quality, while inertia always decreases with more clusters."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 -- Per-Point Silhouette Plot
# ══════════════════════════════════════════════════════════════════════════════
st.header("3. Per-Point Silhouette Plot")

K_selected = st.slider("Select K for silhouette plot", 2, 10, best_k, 1, key="sil_k")

km_sel = KMeans(n_clusters=K_selected, random_state=42, n_init=10)
cluster_labels = km_sel.fit_predict(X_scaled)
sil_values = silhouette_samples(X_scaled, cluster_labels)
mean_sil = silhouette_score(X_scaled, cluster_labels)

st.metric("Overall Silhouette Score", f"{mean_sil:.3f}")

# Build the silhouette plot
fig_sil_plot = go.Figure()
y_lower = 0
colors = px.colors.qualitative.Set2

for i in range(K_selected):
    cluster_sil = sil_values[cluster_labels == i]
    cluster_sil.sort()
    cluster_size = len(cluster_sil)
    y_upper = y_lower + cluster_size

    fig_sil_plot.add_trace(go.Scatter(
        x=cluster_sil,
        y=np.arange(y_lower, y_upper),
        mode="lines",
        fill="tozerox",
        name=f"Cluster {i} (n={cluster_size})",
        line=dict(color=colors[i % len(colors)], width=0.5),
        fillcolor=colors[i % len(colors)],
    ))

    # Label with cluster number
    fig_sil_plot.add_annotation(
        x=-0.05, y=y_lower + cluster_size / 2,
        text=str(i), showarrow=False, font=dict(size=12, color="black")
    )

    y_lower = y_upper + 10  # gap between clusters

# Mean silhouette line
fig_sil_plot.add_vline(
    x=mean_sil, line_dash="dash", line_color="red",
    annotation_text=f"Mean: {mean_sil:.3f}"
)

apply_common_layout(fig_sil_plot, title=f"Silhouette Plot (K={K_selected})", height=550)
fig_sil_plot.update_layout(
    xaxis_title="Silhouette Coefficient",
    yaxis_title="Points (sorted within cluster)",
    yaxis=dict(showticklabels=False),
)
st.plotly_chart(fig_sil_plot, use_container_width=True)

# Per-cluster stats
cluster_stats = []
for i in range(K_selected):
    cluster_sil = sil_values[cluster_labels == i]
    cluster_stats.append({
        "Cluster": i,
        "Size": len(cluster_sil),
        "Mean Silhouette": cluster_sil.mean(),
        "Min Silhouette": cluster_sil.min(),
        "% Negative": (cluster_sil < 0).sum() / len(cluster_sil) * 100,
    })

stats_df = pd.DataFrame(cluster_stats).round(3)
st.dataframe(stats_df, use_container_width=True)

insight_box(
    "Clusters with silhouette values well above the mean are well-defined. Clusters "
    "with many points below 0 contain misassigned points. Uneven widths suggest "
    "imbalanced cluster sizes."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 -- K-Means vs Hierarchical at Different K
# ══════════════════════════════════════════════════════════════════════════════
st.header("4. Comparing K-Means vs Hierarchical Clustering")

st.markdown(
    "Let's compare how K-Means and hierarchical clustering (with different linkage "
    "methods) perform at various K values using the silhouette score."
)

methods_to_compare = {
    "K-Means": None,
    "Ward Linkage": "ward",
    "Complete Linkage": "complete",
    "Average Linkage": "average",
}

comparison_results = []
for k in range(2, 11):
    for name, linkage in methods_to_compare.items():
        if linkage is None:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
        else:
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage)

        labels_comp = model.fit_predict(X_scaled)
        sil_comp = silhouette_score(X_scaled, labels_comp)
        comparison_results.append({
            "K": k,
            "Method": name,
            "Silhouette Score": sil_comp,
        })

comp_df = pd.DataFrame(comparison_results)

fig_comp = px.line(
    comp_df, x="K", y="Silhouette Score", color="Method",
    title="Silhouette Score Comparison: K-Means vs Hierarchical",
    markers=True,
    color_discrete_map={
        "K-Means": "#2E86C1",
        "Ward Linkage": "#E63946",
        "Complete Linkage": "#2A9D8F",
        "Average Linkage": "#FB8500",
    }
)
apply_common_layout(fig_comp, title="Silhouette Score Comparison: K-Means vs Hierarchical", height=450)
fig_comp.update_layout(xaxis=dict(dtick=1))
st.plotly_chart(fig_comp, use_container_width=True)

# Best method at each K
st.subheader("Best Method at Each K")
best_at_k = comp_df.loc[comp_df.groupby("K")["Silhouette Score"].idxmax()]
st.dataframe(best_at_k[["K", "Method", "Silhouette Score"]].round(3), use_container_width=True)

# Overall best
overall_best = comp_df.loc[comp_df["Silhouette Score"].idxmax()]
insight_box(
    f"The best silhouette score overall is **{overall_best['Silhouette Score']:.3f}** "
    f"with **{overall_best['Method']}** at **K = {int(overall_best['K'])}**."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 -- Silhouette with Actual City Labels
# ══════════════════════════════════════════════════════════════════════════════
st.header("5. Silhouette Score with Actual City Labels")

st.markdown(
    "What if we use the actual city assignments as 'clusters'? The silhouette score "
    "tells us how well-separated the cities are in feature space."
)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
city_encoded = le.fit_transform(sample["city"])
sil_actual = silhouette_score(X_scaled, city_encoded)

st.metric("Silhouette Score (Actual Cities)", f"{sil_actual:.3f}")

# Per-city silhouette
sil_by_city_vals = silhouette_samples(X_scaled, city_encoded)
city_sil_stats = []
for i, city in enumerate(le.classes_):
    mask = city_encoded == i
    city_sil = sil_by_city_vals[mask]
    city_sil_stats.append({
        "City": city,
        "Mean Silhouette": city_sil.mean(),
        "Std Silhouette": city_sil.std(),
        "% Negative": (city_sil < 0).sum() / len(city_sil) * 100,
        "Size": len(city_sil),
    })

city_sil_df = pd.DataFrame(city_sil_stats).round(3)

fig_city_sil = px.bar(
    city_sil_df, x="City", y="Mean Silhouette",
    title="Mean Silhouette Score by City",
    color="City", color_discrete_map=CITY_COLORS,
    text_auto=".3f",
    error_y="Std Silhouette"
)
apply_common_layout(fig_city_sil, title="Mean Silhouette Score by City", height=400)
fig_city_sil.update_layout(showlegend=False)
st.plotly_chart(fig_city_sil, use_container_width=True)

st.dataframe(city_sil_df, use_container_width=True)

# Which cities are hardest to separate?
worst_city = city_sil_df.loc[city_sil_df["Mean Silhouette"].idxmin(), "City"]
best_city = city_sil_df.loc[city_sil_df["Mean Silhouette"].idxmax(), "City"]
insight_box(
    f"**{best_city}** has the highest silhouette score -- it is the most distinct. "
    f"**{worst_city}** has the lowest -- its weather overlaps most with other cities. "
    "Texas cities generally have lower silhouette scores because they overlap with each other."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 -- Guidelines for Interpretation
# ══════════════════════════════════════════════════════════════════════════════
st.header("6. Guidelines for Silhouette Score Interpretation")

st.markdown("""
| Silhouette Score | Interpretation |
|-----------------|----------------|
| 0.71 -- 1.00 | Strong structure found |
| 0.51 -- 0.70 | Reasonable structure found |
| 0.26 -- 0.50 | Weak structure; could be artificial |
| <= 0.25 | No substantial structure found |

These are rough guidelines. The best approach is to **compare across methods and K values**.
""")

warning_box(
    "Silhouette scores can be misleading when clusters have very different sizes or "
    "densities. Always combine with other metrics (ARI if labels are available, "
    "elbow method, domain knowledge)."
)

code_example("""
from sklearn.metrics import silhouette_score, silhouette_samples

# Overall silhouette score
score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {score:.3f}")

# Per-point silhouette values (for the silhouette plot)
sample_scores = silhouette_samples(X_scaled, labels)

# Compare methods
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"K={k}: Silhouette = {sil:.3f}")

# Compare K-Means vs Hierarchical
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=6, linkage='ward')
hc_labels = hc.fit_predict(X_scaled)
print(f"Hierarchical: {silhouette_score(X_scaled, hc_labels):.3f}")
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 -- Quiz & Takeaways
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

quiz(
    "A point has silhouette score s(i) = -0.3. What does this mean?",
    [
        "The point is perfectly assigned to its cluster",
        "The point is likely assigned to the wrong cluster (closer to a neighboring cluster)",
        "The point is noise",
        "The clustering algorithm failed",
    ],
    correct_idx=1,
    explanation="A negative silhouette means b(i) < a(i): the point is closer to a different cluster than its own.",
    key="q_sil_1"
)

quiz(
    "You run K-Means with K=2 and get silhouette score 0.6, and K=6 gives 0.35. What should you conclude?",
    [
        "K=2 is always better because it has a higher silhouette score",
        "K=2 has more compact clusters, but K=6 may better reflect the true data structure (6 cities). Context matters.",
        "K=6 is always better because there are 6 cities",
        "Neither is useful because the scores are below 1.0",
    ],
    correct_idx=1,
    explanation="Fewer clusters often gives higher silhouette scores. But domain knowledge (6 cities) matters. The silhouette is one tool among many.",
    key="q_sil_2"
)

takeaways([
    "The silhouette score measures how well each point fits in its cluster vs. the nearest other cluster.",
    "Range: -1 (wrong cluster) to +1 (perfect cluster match). Values near 0 mean boundary points.",
    "Per-point silhouette plots reveal cluster quality at a granular level.",
    "Silhouette score helps compare different algorithms (K-Means vs hierarchical) and choose K.",
    "Lower K often gives higher silhouette scores -- use domain knowledge alongside the metric.",
    "Texas cities have lower per-city silhouette scores because their weather patterns overlap.",
])
