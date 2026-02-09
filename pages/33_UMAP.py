"""Chapter 33 -- UMAP (Uniform Manifold Approximation and Projection)."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(33, "UMAP", part="VII")

st.markdown(
    "If PCA is the sensible accountant and t-SNE is the brilliant but mercurial "
    "artist, UMAP is the engineer who looked at both and said 'I can do this faster "
    "*and* better.' UMAP is a modern nonlinear embedding method that is genuinely "
    "faster than t-SNE, preserves more of the global structure (so inter-cluster "
    "distances actually mean something, sort of), and -- critically -- can transform "
    "new data points that were not in the original training set. It has essentially "
    "replaced t-SNE as the default nonlinear embedding tool in most workflows."
)

# ── Load & filter data ───────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

concept_box(
    "How UMAP Works",
    "UMAP's theory involves the words 'fuzzy simplicial sets' and 'Riemannian "
    "manifold,' which should give you a sense of the mathematical ambition here. "
    "But the practical intuition is approachable: UMAP builds a weighted graph "
    "connecting each point to its k-nearest neighbors in high-dimensional space, "
    "then optimizes a 2-D layout that preserves the structure of that graph as "
    "faithfully as possible. Two knobs control everything: **n_neighbors** "
    "(how many neighbors define 'local' -- higher means more global) and "
    "**min_dist** (how tightly points can cluster together -- lower means denser clumps)."
)

formula_box(
    "UMAP Objective (Cross-Entropy)",
    r"C = \sum_{e \in E} \left[ w_h(e)\,\log\frac{w_h(e)}{w_l(e)} "
    r"+ (1-w_h(e))\,\log\frac{1-w_h(e)}{1-w_l(e)} \right]",
    "w_h = edge weights in the high-D graph, w_l = edge weights in the low-D "
    "layout. Minimizing this cross-entropy makes the 2-D layout match the "
    "high-D neighborhood structure. The repulsive term (second part) is what "
    "prevents everything from collapsing into a single blob."
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.subheader("UMAP Settings")
n_neighbors = st.sidebar.slider("n_neighbors", 5, 200, 15, 5, key="umap_nn")
min_dist = st.sidebar.slider("min_dist", 0.0, 1.0, 0.1, 0.05, key="umap_md")
sample_size = st.sidebar.slider("Sample Size", 500, 5000, 2000, 250, key="umap_sample")

# ── Subsample & scale ────────────────────────────────────────────────────────
sub = fdf.dropna(subset=FEATURE_COLS).sample(
    n=min(sample_size, len(fdf)), random_state=42
)
X = sub[FEATURE_COLS].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Fit UMAP ─────────────────────────────────────────────────────────────────
st.header("1. UMAP Embedding")

try:
    import umap

    with st.spinner(f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})..."):
        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors,
            min_dist=min_dist, random_state=42,
        )
        X_umap = reducer.fit_transform(X_scaled)

    emb = pd.DataFrame({
        "UMAP 1": X_umap[:, 0],
        "UMAP 2": X_umap[:, 1],
        "city": sub["city"].values,
        "season": sub["season"].values,
    })

    color_by = st.radio("Color by:", ["city", "season"], horizontal=True, key="umap_color")

    if color_by == "city":
        fig = px.scatter(
            emb, x="UMAP 1", y="UMAP 2", color="city",
            color_discrete_map=CITY_COLORS, opacity=0.5,
            title=f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})",
        )
    else:
        fig = px.scatter(
            emb, x="UMAP 1", y="UMAP 2", color="season",
            color_discrete_sequence=["#2E86C1", "#2A9D8F", "#E63946", "#F4A261"],
            opacity=0.5,
            title=f"UMAP by Season (n_neighbors={n_neighbors}, min_dist={min_dist})",
        )
    apply_common_layout(fig, height=600)
    st.plotly_chart(fig, use_container_width=True)

    umap_available = True

except ImportError:
    st.error(
        "The `umap-learn` package is not installed. "
        "Install it with `pip install umap-learn` to enable this chapter."
    )
    umap_available = False

insight_box(
    "**n_neighbors** is UMAP's version of t-SNE's perplexity: small values "
    "mean 'focus on the immediate neighborhood' (tight clusters), large values "
    "mean 'zoom out and consider the broader topology.' **min_dist** is the "
    "knob that controls how densely things clump -- set it to 0 and you get "
    "extremely dense blobs; push it toward 1 and everything spreads out into a "
    "more uniform distribution. Together, they give you surprisingly fine control "
    "over what the embedding looks like."
)

# ── Section 2: Effect of Parameters ──────────────────────────────────────────
st.header("2. Effect of n_neighbors and min_dist")

st.markdown(
    "Let me save you some trial-and-error by showing what happens when you twiddle "
    "the two main knobs. Below, we hold min_dist fixed and vary n_neighbors, "
    "then hold n_neighbors fixed and vary min_dist. You can also just play with "
    "the sidebar sliders and watch the main plot update in real time."
)

if umap_available:
    comp_sample = sub.sample(n=min(1200, len(sub)), random_state=7)
    X_comp = StandardScaler().fit_transform(comp_sample[FEATURE_COLS].values)

    nn_values = [5, 30, 100]
    cols = st.columns(3)
    for idx, nn in enumerate(nn_values):
        with cols[idx]:
            st.subheader(f"n_neighbors={nn}")
            r = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=0.1, random_state=42)
            Xr = r.fit_transform(X_comp)
            rdf = pd.DataFrame({
                "x": Xr[:, 0], "y": Xr[:, 1],
                "city": comp_sample["city"].values,
            })
            fig_nn = px.scatter(
                rdf, x="x", y="y", color="city",
                color_discrete_map=CITY_COLORS, opacity=0.5,
            )
            fig_nn.update_layout(
                showlegend=False, height=350,
                margin=dict(t=10, b=10, l=10, r=10),
                template="plotly_white",
            )
            st.plotly_chart(fig_nn, use_container_width=True)

    st.markdown("---")
    st.subheader("Effect of min_dist")
    md_values = [0.0, 0.25, 0.8]
    cols2 = st.columns(3)
    for idx, md in enumerate(md_values):
        with cols2[idx]:
            st.subheader(f"min_dist={md}")
            r2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=md, random_state=42)
            Xr2 = r2.fit_transform(X_comp)
            rdf2 = pd.DataFrame({
                "x": Xr2[:, 0], "y": Xr2[:, 1],
                "city": comp_sample["city"].values,
            })
            fig_md = px.scatter(
                rdf2, x="x", y="y", color="city",
                color_discrete_map=CITY_COLORS, opacity=0.5,
            )
            fig_md.update_layout(
                showlegend=False, height=350,
                margin=dict(t=10, b=10, l=10, r=10),
                template="plotly_white",
            )
            st.plotly_chart(fig_md, use_container_width=True)

warning_box(
    "Like t-SNE, you should not over-interpret UMAP cluster sizes or exact "
    "distances. However -- and this is a meaningful 'however' -- UMAP preserves "
    "more of the global structure than t-SNE does. So if cluster A is far from "
    "cluster B in the UMAP plot, there is a *better* (though still imperfect) "
    "chance that they are genuinely dissimilar in the original space."
)

code_example("""
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# UMAP can also transform new data (unlike t-SNE):
X_new_umap = reducer.transform(X_new_scaled)
""")

# ── Section 3: PCA vs t-SNE vs UMAP Side-by-Side ────────────────────────────
st.header("3. PCA vs t-SNE vs UMAP -- Side-by-Side Comparison")

st.markdown(
    "Alright, the moment of truth. We throw all three methods at the same data and "
    "see what each one finds. This is the dimensionality-reduction equivalent of a "
    "taste test, and it is genuinely instructive."
)

comp_sub = sub.sample(n=min(1500, len(sub)), random_state=99)
Xc = StandardScaler().fit_transform(comp_sub[FEATURE_COLS].values)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(Xc)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000, init="pca")
X_tsne = tsne.fit_transform(Xc)

# UMAP
if umap_available:
    um = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_um = um.fit_transform(Xc)

methods = ["PCA", "t-SNE", "UMAP"] if umap_available else ["PCA", "t-SNE"]
embeddings = [X_pca, X_tsne] + ([X_um] if umap_available else [])

cols3 = st.columns(len(methods))
for idx, (name, Xemb) in enumerate(zip(methods, embeddings)):
    with cols3[idx]:
        edf = pd.DataFrame({
            "Dim 1": Xemb[:, 0], "Dim 2": Xemb[:, 1],
            "city": comp_sub["city"].values,
        })
        figc = px.scatter(
            edf, x="Dim 1", y="Dim 2", color="city",
            color_discrete_map=CITY_COLORS, opacity=0.5, title=name,
        )
        figc.update_layout(
            showlegend=(idx == 0), height=400,
            margin=dict(t=40, b=10, l=10, r=10),
            template="plotly_white",
        )
        st.plotly_chart(figc, use_container_width=True)

# Summary table
st.subheader("Method Comparison")
comparison = pd.DataFrame({
    "Property": [
        "Type", "Speed", "Global Structure",
        "Local Structure", "New Data Transform", "Use Case",
    ],
    "PCA": [
        "Linear", "Very fast", "Preserved",
        "Limited", "Yes", "Fast overview, preprocessing",
    ],
    "t-SNE": [
        "Nonlinear", "Slow", "Not preserved",
        "Well preserved", "No", "Visualization of clusters",
    ],
    "UMAP": [
        "Nonlinear", "Fast", "Partially preserved",
        "Well preserved", "Yes", "Visualization + general DR",
    ],
})
st.dataframe(comparison, use_container_width=True, hide_index=True)

insight_box(
    "UMAP is, in most practical scenarios, the best of both worlds: nearly as "
    "good at separating clusters as t-SNE, much faster, and capable of transforming "
    "new data. PCA remains the go-to for linear preprocessing before feeding data into "
    "ML models, because its axes are interpretable and the transform is dirt cheap. "
    "The honest answer to 'which should I use?' is: PCA first (always), then UMAP "
    "if you want a beautiful nonlinear visualization."
)

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "Which advantage does UMAP have over t-SNE?",
    [
        "UMAP is a linear method",
        "UMAP can transform new unseen data points",
        "UMAP does not require scaling",
        "UMAP always produces the same result without a random seed",
    ],
    1,
    "UMAP learns an actual mapping function, so you can call reducer.transform() "
    "on new data that was not part of the original fit. t-SNE has no such ability -- "
    "you would need to re-run the entire fitting process including the new points.",
    key="umap_quiz",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "UMAP preserves both local and some global structure by optimizing a fuzzy topological graph representation -- which is a fancy way of saying it respects neighborhoods.",
    "**n_neighbors** controls local vs global emphasis (like perplexity in t-SNE). This is the single most important parameter to experiment with.",
    "**min_dist** controls how tightly clusters are packed. 0 = dense blobs, 1 = uniform spread.",
    "UMAP is faster than t-SNE and supports transforming new data, which makes it strictly more useful in production settings.",
    "For a full picture, compare PCA, t-SNE, and UMAP side-by-side on the same subsample. Each method shows you something the others miss.",
])
