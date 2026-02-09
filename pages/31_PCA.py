"""Chapter 31 – Principal Component Analysis (PCA)."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, multi_subplot
from utils.constants import CITY_COLORS, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(31, "Principal Component Analysis (PCA)", part="VII")

st.markdown(
    "Imagine you have four numbers describing the weather every hour -- temperature, "
    "humidity, wind speed, and pressure -- and you want to plot them all on a single "
    "flat screen. The problem is that a flat screen only has two dimensions, and you "
    "have four. PCA is the mathematically principled way to squish those four dimensions "
    "down to two while losing as little information as possible. It finds the directions "
    "in your data where the most *action* is happening and projects everything onto those. "
    "Think of it like finding out that all of Shakespeare's plays can mostly be described "
    "along two axes: 'comedic vs tragic' and 'Italian setting vs English setting.' "
    "We are about to do the same thing with weather."
)

# ── Load & filter data ───────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

concept_box(
    "What Is PCA?",
    "PCA is, at heart, a rotation. You take your data, find the direction along which "
    "it varies the most (that becomes your first axis, PC1), then find the next most "
    "variable direction that is perpendicular to the first (PC2), and so on. "
    "Mathematically, you are computing the eigenvectors of the covariance matrix, "
    "which sounds intimidating but really just means 'find the directions the data "
    "stretches the most.' The beautiful thing is that this is a lossless rotation -- "
    "you only lose information when you decide to *drop* the less important axes."
)

formula_box(
    "Covariance Matrix Eigen-Decomposition",
    r"\Sigma = V \Lambda V^T",
    "V contains the eigenvectors (the new axes, i.e., principal directions) and "
    "Lambda contains the eigenvalues, which tell you how much variance lives along "
    "each direction. Bigger eigenvalue = more interesting axis."
)

# ── Subsample for speed ──────────────────────────────────────────────────────
st.sidebar.subheader("PCA Settings")
n_components = st.sidebar.slider("Number of Components", 1, 4, 2, key="pca_n")
sample_size = st.sidebar.slider("Sample Size", 1000, 10000, 5000, 500, key="pca_sample")

sub = fdf.dropna(subset=FEATURE_COLS).sample(n=min(sample_size, len(fdf)), random_state=42)
X = sub[FEATURE_COLS].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Fit PCA ──────────────────────────────────────────────────────────────────
pca_full = PCA(n_components=4)
pca_full.fit(X_scaled)

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# ── Section 1: Explained Variance ────────────────────────────────────────────
st.header("1. Explained Variance Ratio")

st.markdown(
    "So we have rotated our data. The natural question is: how much did we keep? "
    "Each principal component explains some fraction of the total variance. The first "
    "component is always the greediest -- it hogs as much variance as it can. The "
    "second gets whatever is left over in the best remaining direction, and so on. "
    "The cumulative curve below tells you how many axes you need before you have "
    "captured 'most' of the story."
)

evr = pca_full.explained_variance_ratio_
cum_evr = np.cumsum(evr)
evr_df = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(4)],
    "Explained Variance Ratio": evr,
    "Cumulative": cum_evr,
})

fig_evr = go.Figure()
fig_evr.add_trace(go.Bar(
    x=evr_df["Component"], y=evr_df["Explained Variance Ratio"],
    name="Individual", marker_color="#2E86C1",
))
fig_evr.add_trace(go.Scatter(
    x=evr_df["Component"], y=evr_df["Cumulative"],
    name="Cumulative", mode="lines+markers", marker_color="#E63946",
))
fig_evr.update_layout(yaxis_title="Variance Ratio", yaxis_range=[0, 1.05])
apply_common_layout(fig_evr, "Explained Variance by Component", 450)
st.plotly_chart(fig_evr, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("PC1 Variance", f"{evr[0]:.1%}")
col2.metric("PC1 + PC2", f"{cum_evr[1]:.1%}")
col3.metric("All 4 PCs", f"{cum_evr[3]:.1%}")

insight_box(
    f"The first two components explain **{cum_evr[1]:.1%}** of total variance. "
    "In plain English: a 2-D scatter plot captures most of the interesting structure "
    "in our 4 weather features. The remaining components are not *useless*, "
    "but they are more like footnotes than chapters."
)

code_example("""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)
""")

# ── Section 2: 2-D Scatter (Projection) ─────────────────────────────────────
st.header("2. 2-D PCA Projection")

st.markdown(
    "Here is the payoff. Each point below is one hourly weather observation, "
    "projected from 4-D space down to the first two principal components. "
    "We color by city (or season) to see whether PCA organizes things in a way "
    "that makes climatic sense. If Los Angeles and Houston end up on opposite "
    "sides of the plot, PCA has successfully discovered that these cities have "
    "very different weather fingerprints -- without us telling it anything about geography."
)

proj = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1] if n_components >= 2 else 0,
    "city": sub["city"].values,
    "season": sub["season"].values,
})

color_by = st.radio("Color by:", ["city", "season"], horizontal=True, key="pca_color")

if color_by == "city":
    fig_proj = px.scatter(
        proj, x="PC1", y="PC2", color="city",
        color_discrete_map=CITY_COLORS, opacity=0.4,
        title="Weather Observations in PCA Space",
    )
else:
    fig_proj = px.scatter(
        proj, x="PC1", y="PC2", color="season", opacity=0.4,
        color_discrete_sequence=["#2E86C1", "#2A9D8F", "#E63946", "#F4A261"],
        title="Weather Observations in PCA Space (by Season)",
    )
apply_common_layout(fig_proj, height=550)
st.plotly_chart(fig_proj, use_container_width=True)

insight_box(
    "Cities with very different climates (say, Los Angeles vs Houston) tend to separate "
    "along PC1 because their temperature and humidity profiles differ the most -- and "
    "those are exactly the features carrying the most variance. PCA is not psychic; "
    "it is just very good at finding the axis along which things *disagree* the most."
)

# ── Section 3: Biplot ────────────────────────────────────────────────────────
st.header("3. Biplot -- Feature Contributions")

st.markdown(
    "A **biplot** overlays the original feature directions (as arrows) onto the "
    "PCA scatter. This answers the question 'what does PC1 actually *mean*?' "
    "Longer arrows mean that feature contributes more to the spread. The direction "
    "tells you which component it loads on. If the temperature arrow points strongly "
    "along PC1, that means PC1 is basically 'the temperature axis, plus friends.'"
)

loadings = pca_full.components_[:2]  # (2, 4)
scale_factor = 3.0

fig_bi = px.scatter(
    proj, x="PC1", y="PC2", color="city",
    color_discrete_map=CITY_COLORS, opacity=0.25,
    title="Biplot: Feature Arrows on PCA Projection",
)

feature_names = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS]
for i, feat in enumerate(feature_names):
    fig_bi.add_annotation(
        ax=0, ay=0, axref="x", ayref="y",
        x=loadings[0, i] * scale_factor,
        y=loadings[1, i] * scale_factor,
        showarrow=True,
        arrowhead=3, arrowsize=1.5, arrowwidth=2,
        arrowcolor="#1B4F72",
    )
    fig_bi.add_annotation(
        x=loadings[0, i] * scale_factor * 1.15,
        y=loadings[1, i] * scale_factor * 1.15,
        text=feat, showarrow=False,
        font=dict(size=12, color="#1B4F72"),
    )

apply_common_layout(fig_bi, height=600)
st.plotly_chart(fig_bi, use_container_width=True)

# Loadings table
st.subheader("Component Loadings Table")
loadings_df = pd.DataFrame(
    pca_full.components_,
    columns=[FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
    index=[f"PC{i+1}" for i in range(4)],
).round(3)
st.dataframe(loadings_df, use_container_width=True)

warning_box(
    "PCA loadings depend *heavily* on scaling. If you skip StandardScaler, "
    "surface_pressure_hpa (values around 1000) would dominate simply because "
    "its numbers are bigger, not because it carries more information. This is "
    "the classic 'comparing meters to kilograms' problem, and it will silently "
    "ruin your analysis if you forget."
)

# ── Section 4: Reconstruction Error ─────────────────────────────────────────
st.header("4. Reconstruction Error")

st.markdown(
    "Here is an important thought experiment. If we project our 4-D data down "
    "to 2-D and then try to reverse the process -- project back up to 4-D -- "
    "how much detail do we lose? The **reconstruction error** quantifies exactly "
    "this. It is the gap between what you started with and what you get back "
    "after the round trip through PCA-land."
)

errors = []
for k in range(1, 5):
    pca_k = PCA(n_components=k)
    transformed = pca_k.fit_transform(X_scaled)
    reconstructed = pca_k.inverse_transform(transformed)
    mse = np.mean((X_scaled - reconstructed) ** 2)
    errors.append({"Components": k, "MSE": mse})

err_df = pd.DataFrame(errors)
fig_err = px.bar(
    err_df, x="Components", y="MSE",
    title="Reconstruction Error vs Number of Components",
    labels={"MSE": "Mean Squared Error"},
    color_discrete_sequence=["#2E86C1"],
)
fig_err.update_xaxes(dtick=1)
apply_common_layout(fig_err, height=400)
st.plotly_chart(fig_err, use_container_width=True)

# Show a sample reconstruction
st.subheader("Sample Reconstruction")
st.markdown(
    f"With **{n_components} component(s)**, here is the round-trip reconstruction "
    "for the first 5 observations (in scaled units). You can flip between tabs to "
    "see how close they are -- and more importantly, where they diverge."
)
X_recon = pca.inverse_transform(X_pca)
orig_df = pd.DataFrame(X_scaled[:5], columns=FEATURE_COLS).round(3)
recon_df = pd.DataFrame(X_recon[:5], columns=FEATURE_COLS).round(3)

tab1, tab2 = st.tabs(["Original (Scaled)", "Reconstructed"])
with tab1:
    st.dataframe(orig_df, use_container_width=True)
with tab2:
    st.dataframe(recon_df, use_container_width=True)

insight_box(
    "With 2 components we lose some fine detail, but the reconstruction is usually "
    "close enough for visualization purposes. For downstream ML tasks, the "
    "explained-variance curve is your guide: pick the number of components where "
    "the cumulative line starts to plateau, and you are probably keeping everything "
    "that matters."
)

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "If the first two principal components explain 85% of variance, what does "
    "the remaining 15% represent?",
    [
        "Noise that should always be discarded",
        "Information captured by PC3 and PC4",
        "The mean of the data",
        "Outliers only",
    ],
    1,
    "The remaining 15% is real information living in the less important directions. "
    "Whether you should keep it depends entirely on your task. For visualization, "
    "you can usually ignore it. For a high-stakes ML model, maybe not.",
    key="pca_quiz",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "PCA finds orthogonal directions of maximum variance by computing eigenvectors of the covariance matrix. It is a rotation, not a magic trick.",
    "Always **standardize** features before PCA. If you don't, whichever feature has the biggest numbers wins by default, and that is almost never what you want.",
    "The explained-variance curve is your friend for deciding how many components to keep -- look for where it plateaus.",
    "Biplots answer the crucial interpretive question: 'What do these abstract axes actually mean in terms of my original features?'",
    "Reconstruction error quantifies exactly how much information you sacrificed for the sake of fewer dimensions.",
])
