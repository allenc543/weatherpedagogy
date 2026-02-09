"""Chapter 32 -- t-SNE (t-distributed Stochastic Neighbor Embedding)."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
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
chapter_header(32, "t-SNE", part="VII")

st.markdown(
    "PCA is a nice, well-behaved linear method, which is both its strength and its "
    "limitation. It can only find structure that lives along straight lines. "
    "t-SNE is what you reach for when you suspect the interesting structure is "
    "*nonlinear* -- when clusters are curving around each other in high-dimensional "
    "space in ways that a simple rotation cannot untangle. Think of it as PCA's "
    "wilder, more artistic cousin: it produces gorgeous visualizations of cluster "
    "structure, but it comes with some important caveats that we will get into."
)

# ── Load & filter data ───────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

concept_box(
    "How t-SNE Works",
    "t-SNE does something conceptually elegant. First, it measures how close "
    "every pair of points is in the original high-dimensional space, converting "
    "distances into probabilities (using a Gaussian distribution). Then it tries "
    "to arrange points in 2-D so that the pairwise probabilities (now using a "
    "heavier-tailed Student-t distribution) match as closely as possible. "
    "It minimizes the mismatch using gradient descent. The result preserves "
    "**local** structure beautifully -- nearby points stay nearby -- but it can "
    "distort global distances."
)

formula_box(
    "KL Divergence Objective",
    r"C = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}",
    "p_ij = pairwise similarities in high-D (Gaussian kernel), q_ij = pairwise "
    "similarities in low-D (Student-t with 1 degree of freedom). The heavy tail "
    "of the Student-t is what prevents the 'crowding problem' -- it gives points "
    "more room to spread out in 2-D."
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.subheader("t-SNE Settings")
perplexity = st.sidebar.slider("Perplexity", 5, 100, 30, 5, key="tsne_perp")
sample_size = st.sidebar.slider("Sample Size", 500, 5000, 2000, 250, key="tsne_sample")
color_by = st.sidebar.selectbox(
    "Color By", ["city", "season", "hour"], key="tsne_color"
)

# ── Subsample & scale ────────────────────────────────────────────────────────
sub = fdf.dropna(subset=FEATURE_COLS).sample(
    n=min(sample_size, len(fdf)), random_state=42
)
X = sub[FEATURE_COLS].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Fit t-SNE ────────────────────────────────────────────────────────────────
st.header("1. t-SNE Embedding")

with st.spinner(f"Running t-SNE with perplexity={perplexity}..."):
    tsne = TSNE(
        n_components=2, perplexity=perplexity,
        random_state=42, n_iter=1000, init="pca",
    )
    X_tsne = tsne.fit_transform(X_scaled)

emb = pd.DataFrame({
    "t-SNE 1": X_tsne[:, 0],
    "t-SNE 2": X_tsne[:, 1],
    "city": sub["city"].values,
    "season": sub["season"].values,
    "hour": sub["hour"].values,
})

if color_by == "city":
    fig = px.scatter(
        emb, x="t-SNE 1", y="t-SNE 2", color="city",
        color_discrete_map=CITY_COLORS, opacity=0.5,
        title=f"t-SNE Embedding (perplexity={perplexity})",
    )
elif color_by == "season":
    fig = px.scatter(
        emb, x="t-SNE 1", y="t-SNE 2", color="season",
        color_discrete_sequence=["#2E86C1", "#2A9D8F", "#E63946", "#F4A261"],
        opacity=0.5,
        title=f"t-SNE Embedding colored by Season (perplexity={perplexity})",
    )
else:
    fig = px.scatter(
        emb, x="t-SNE 1", y="t-SNE 2", color="hour",
        color_continuous_scale="twilight", opacity=0.5,
        title=f"t-SNE Embedding colored by Hour (perplexity={perplexity})",
    )
apply_common_layout(fig, height=600)
st.plotly_chart(fig, use_container_width=True)

insight_box(
    "When colored by **city**, you may see distinct clusters for places with very "
    "different climates -- LA's dry mildness vs Houston's swampy warmth. The truly "
    "fun thing is to switch to **hour** coloring: you will often see a gradient "
    "within each cluster, because night observations and day observations have "
    "measurably different feature signatures, even within the same city."
)

# ── Section 2: Perplexity Comparison ─────────────────────────────────────────
st.header("2. Effect of Perplexity")

st.markdown(
    "**Perplexity** is the parameter that makes t-SNE both powerful and maddening. "
    "It roughly controls how many neighbors each point 'pays attention to.' "
    "Low perplexity (5) means each point only cares about its very closest "
    "neighbors, producing tight, fragmented clusters. High perplexity (100) "
    "means each point considers a broader neighborhood, revealing more global "
    "patterns but potentially smearing out fine structure. There is no universally "
    "'right' value -- you are supposed to try several and compare, which is what "
    "we do below."
)

perp_values = [5, 30, 100]
comp_sample = sub.sample(n=min(1500, len(sub)), random_state=7)
X_comp = StandardScaler().fit_transform(comp_sample[FEATURE_COLS].values)

cols = st.columns(3)
for idx, pv in enumerate(perp_values):
    with cols[idx]:
        st.subheader(f"Perplexity = {pv}")
        tsne_c = TSNE(
            n_components=2, perplexity=pv, random_state=42,
            n_iter=1000, init="pca",
        )
        Xc = tsne_c.fit_transform(X_comp)
        comp_df = pd.DataFrame({
            "x": Xc[:, 0], "y": Xc[:, 1],
            "city": comp_sample["city"].values,
        })
        fig_c = px.scatter(
            comp_df, x="x", y="y", color="city",
            color_discrete_map=CITY_COLORS, opacity=0.5,
        )
        fig_c.update_layout(
            showlegend=False, height=350, margin=dict(t=10, b=10, l=10, r=10),
            template="plotly_white",
        )
        st.plotly_chart(fig_c, use_container_width=True)

warning_box(
    "This is the part where I have to be the responsible adult. t-SNE results "
    "change with both the random seed and the perplexity. The axes are meaningless "
    "(there is no 't-SNE 1 is temperature' interpretation). Cluster *distances* "
    "and *sizes* in the plot are unreliable. The only thing you can safely interpret "
    "is whether clusters *exist*. Do not write a paper claiming 'Houston is twice "
    "as far from LA as from Dallas' based on a t-SNE plot."
)

code_example("""
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X_scaled)
""")

# ── Section 3: t-SNE vs PCA ─────────────────────────────────────────────────
st.header("3. t-SNE vs PCA")

st.markdown(
    "You are probably wondering: if PCA already exists, why do we need t-SNE? "
    "The short answer is that PCA is fast and preserves the big-picture global "
    "structure (which dimensions vary most), but it cannot detect nonlinear "
    "clusters -- everything comes out as a continuous blob. t-SNE is slower "
    "and nondeterministic, but it can pull apart clusters that PCA merely "
    "smears together. Below we run both on the same subsample so you can "
    "see the difference firsthand."
)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_comp)

pca_df = pd.DataFrame({
    "Dim 1": X_pca[:, 0], "Dim 2": X_pca[:, 1],
    "city": comp_sample["city"].values,
})
tsne_30 = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000, init="pca")
X_t30 = tsne_30.fit_transform(X_comp)
tsne_df = pd.DataFrame({
    "Dim 1": X_t30[:, 0], "Dim 2": X_t30[:, 1],
    "city": comp_sample["city"].values,
})

c1, c2 = st.columns(2)
with c1:
    fig_pca = px.scatter(
        pca_df, x="Dim 1", y="Dim 2", color="city",
        color_discrete_map=CITY_COLORS, opacity=0.5, title="PCA",
    )
    apply_common_layout(fig_pca, height=400)
    st.plotly_chart(fig_pca, use_container_width=True)
with c2:
    fig_tsne = px.scatter(
        tsne_df, x="Dim 1", y="Dim 2", color="city",
        color_discrete_map=CITY_COLORS, opacity=0.5, title="t-SNE (perp=30)",
    )
    apply_common_layout(fig_tsne, height=400)
    st.plotly_chart(fig_tsne, use_container_width=True)

insight_box(
    "PCA spreads data along the directions of highest variance, which often looks "
    "like a continuous cloud with overlapping cities. t-SNE pulls apart clusters "
    "that were only vaguely separated in PCA space, making them visually distinct. "
    "The trade-off is real though: PCA's axes mean something; t-SNE's do not."
)

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "What does the perplexity parameter in t-SNE control?",
    [
        "The number of output dimensions",
        "The balance between local and global structure (effective number of neighbors)",
        "The learning rate for gradient descent",
        "The number of iterations to run",
    ],
    1,
    "Perplexity is roughly the effective number of nearest neighbors each point "
    "considers. Low values produce tight local clusters, high values reveal broader "
    "patterns. It is one of those parameters where the correct answer is always "
    "'try a few values and see what happens.'",
    key="tsne_quiz",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "t-SNE is a nonlinear method that excels at preserving **local** neighborhood structure -- nearby points in high-D stay nearby in 2-D.",
    "Perplexity controls the local-vs-global trade-off. There is no universally correct value; try several.",
    "Results vary with the random seed, so always set `random_state` for reproducibility and do not panic when re-runs look different.",
    "Do NOT interpret inter-cluster distances or cluster sizes literally. The existence of clusters is meaningful; their relative positions are not.",
    "t-SNE is for **visualization only** -- you cannot use a fitted t-SNE to transform new data points. If you need that, look at UMAP (next chapter).",
])
