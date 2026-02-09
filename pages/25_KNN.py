"""Chapter 25: K-Nearest Neighbors -- Distance, scaling, and the curse of dimensionality."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.ml_helpers import prepare_classification_data, classification_metrics, plot_confusion_matrix
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(25, "K-Nearest Neighbors (KNN)", part="V")
st.markdown(
    "KNN is the algorithm you would invent if you had never heard of machine learning. "
    "Someone asks 'what city does this weather come from?' and you respond: 'let me "
    "find the 5 most similar weather observations in my dataset and see what city "
    "they came from.' That is it. That is the whole algorithm. There is no training "
    "phase, no learned parameters, no model file. The training data *is* the model. "
    "It is absurdly simple, and as we will see, it works surprisingly well -- as long "
    "as you remember to **scale your features**."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
filt = sidebar_filters(df)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 -- How KNN Works
# ══════════════════════════════════════════════════════════════════════════════
st.header("1. How KNN Works")

concept_box(
    "Lazy Learning",
    "KNN is what ML people call a <b>lazy learner</b>, and they do not mean it as an "
    "insult. It just means the algorithm does zero work during training -- it literally "
    "stores the entire dataset and waits. All the computation happens at prediction "
    "time:<br>"
    "1. Compute the distance from the new point to <i>every</i> training point<br>"
    "2. Find the K closest neighbors<br>"
    "3. Let them vote -- the majority class wins<br><br>"
    "This is great for small datasets. For large ones, the 'compute distance to "
    "everything' part becomes a problem."
)

formula_box(
    "Euclidean Distance",
    r"\underbrace{d(\mathbf{x}, \mathbf{x}')}_{\text{distance between readings}} = \sqrt{\underbrace{\sum_{j=1}^{p}}_{\text{over all features}} \underbrace{(x_j - x_j')^2}_{\text{squared feature diff}}}",
    "The most common distance metric. But notice something crucial: this formula treats "
    "all features equally. If one feature ranges from 0 to 1000 and another from 0 to 1, "
    "the big feature will completely dominate the distance calculation. This is about to "
    "matter a lot."
)

st.markdown("""
**Other distance metrics:**
- **Manhattan distance**: sum of absolute differences (good for grid-like data)
- **Minkowski distance**: generalization of both Euclidean and Manhattan
- **Cosine similarity**: measures angle between vectors (ignores magnitude)
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 -- Why Scaling Matters (Unscaled vs Scaled)
# ══════════════════════════════════════════════════════════════════════════════
st.header("2. Why Feature Scaling is Crucial for KNN")

concept_box(
    "The Scale Problem",
    "KNN with unscaled features is like measuring distance on a map where one inch "
    "represents a mile going north but a thousand miles going east. Temperature ranges "
    "from about -10 to 40 (degrees C), while surface pressure ranges from about 980 "
    "to 1030 (hPa). Without scaling, pressure values are roughly <b>30 times larger</b> "
    "than temperature values, so the Euclidean distance is dominated almost entirely "
    "by pressure. The algorithm is essentially ignoring temperature, humidity, and wind "
    "speed. It is classifying cities based on barometric pressure alone, which is... "
    "not great."
)

# Prepare data -- unscaled
le_comp = LabelEncoder()
X_raw = filt[FEATURE_COLS].dropna()
y_raw = le_comp.fit_transform(filt.loc[X_raw.index, "city"])
X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

# Unscaled KNN
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_tr_raw, y_tr)
acc_unscaled = knn_unscaled.score(X_te_raw, y_te)

# Scaled KNN
scaler_comp = StandardScaler()
X_tr_scaled = scaler_comp.fit_transform(X_tr_raw)
X_te_scaled = scaler_comp.transform(X_te_raw)
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_tr_scaled, y_tr)
acc_scaled = knn_scaled.score(X_te_scaled, y_te)

col1, col2 = st.columns(2)
with col1:
    st.metric("Unscaled Accuracy (K=5)", f"{acc_unscaled:.1%}")
    st.caption("Features in original units -- pressure dominates everything")
with col2:
    st.metric("Scaled Accuracy (K=5)", f"{acc_scaled:.1%}", delta=f"+{acc_scaled - acc_unscaled:.1%}")
    st.caption("Features standardized (mean=0, std=1) -- all features contribute equally")

# Show feature ranges
range_df = pd.DataFrame({
    "Feature": [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
    "Min": [filt[f].min() for f in FEATURE_COLS],
    "Max": [filt[f].max() for f in FEATURE_COLS],
    "Range": [filt[f].max() - filt[f].min() for f in FEATURE_COLS],
    "Std Dev": [filt[f].std() for f in FEATURE_COLS],
}).round(2)
st.dataframe(range_df, use_container_width=True)

warning_box(
    f"Look at these ranges. Surface pressure has a range of ~{range_df.iloc[3]['Range']:.0f} hPa "
    f"while temperature has a range of ~{range_df.iloc[0]['Range']:.0f} degrees C. When you compute "
    "Euclidean distance, the pressure difference alone is so large that it drowns out "
    "everything else. The model cannot even hear temperature over the noise of pressure. "
    "Scaling fixes this completely."
)

# Bar chart showing the difference
scale_comp = pd.DataFrame({
    "Condition": ["Unscaled", "Scaled"],
    "Accuracy": [acc_unscaled, acc_scaled],
})
fig_scale = px.bar(
    scale_comp, x="Condition", y="Accuracy",
    title="KNN Accuracy: Unscaled vs Scaled Features",
    color="Condition", color_discrete_map={"Unscaled": "#E63946", "Scaled": "#2E86C1"},
    text_auto=".1%"
)
apply_common_layout(fig_scale, title="KNN Accuracy: Unscaled vs Scaled Features", height=350)
fig_scale.update_layout(showlegend=False)
st.plotly_chart(fig_scale, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 -- Choosing K
# ══════════════════════════════════════════════════════════════════════════════
st.header("3. Choosing K: The Bias-Variance Trade-off")

concept_box(
    "K and Model Complexity",
    "The choice of K is KNN's one real hyperparameter, and it controls a beautiful "
    "bias-variance trade-off:<br><br>"
    "- <b>K = 1</b>: The most complex model possible. Every prediction depends on the "
    "single nearest neighbor. Training accuracy is trivially 100% (each point is its "
    "own nearest neighbor). But it overfits terribly -- it memorizes every noise point.<br>"
    "- <b>Large K</b>: Smoother, more conservative predictions. At the extreme (K = N, "
    "the entire dataset), every point is predicted as the majority class regardless of "
    "its features. Underfitting city."
)

K_val = st.slider("Choose K (number of neighbors)", 1, 51, 5, 2, key="knn_k")

# Use scaled data from here on
X_train_s, X_test_s, y_train_s, y_test_s, le_s, scaler_s = prepare_classification_data(
    filt, FEATURE_COLS, target="city", test_size=0.2, scale=True, seed=42
)
labels = le_s.classes_.tolist()

knn_model = KNeighborsClassifier(n_neighbors=K_val)
knn_model.fit(X_train_s, y_train_s)
y_pred_knn = knn_model.predict(X_test_s)
metrics_knn = classification_metrics(y_test_s, y_pred_knn, labels=labels)

col1, col2, col3 = st.columns(3)
col1.metric("K", K_val)
col2.metric("Test Accuracy", f"{metrics_knn['accuracy']:.1%}")
col3.metric("Train Accuracy", f"{knn_model.score(X_train_s, y_train_s):.1%}")

st.plotly_chart(
    plot_confusion_matrix(metrics_knn["confusion_matrix"], labels),
    use_container_width=True
)

# K sweep
st.subheader("Accuracy vs K")
k_range = list(range(1, 52, 2))
k_train_accs = []
k_test_accs = []
for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_s, y_train_s)
    k_train_accs.append(knn_temp.score(X_train_s, y_train_s))
    k_test_accs.append(knn_temp.score(X_test_s, y_test_s))

k_df = pd.DataFrame({
    "K": k_range,
    "Train Accuracy": k_train_accs,
    "Test Accuracy": k_test_accs,
})
k_melt = k_df.melt(id_vars="K", var_name="Set", value_name="Accuracy")

fig_k = px.line(
    k_melt, x="K", y="Accuracy", color="Set",
    title="Train vs Test Accuracy by K",
    markers=True,
    color_discrete_map={"Train Accuracy": "#E63946", "Test Accuracy": "#2E86C1"},
)
apply_common_layout(fig_k, title="Train vs Test Accuracy by K", height=400)
st.plotly_chart(fig_k, use_container_width=True)

best_k = k_df.loc[k_df["Test Accuracy"].idxmax(), "K"]
best_k_acc = k_df["Test Accuracy"].max()
insight_box(
    f"The best test accuracy is **{best_k_acc:.1%}** at **K = {int(best_k)}**. "
    "Look at K=1: training accuracy is a perfect 100%, because every point is its own "
    "nearest neighbor (obviously). But test accuracy is lower, because the model is "
    "memorizing noise. As K increases, the model smooths out, trading training "
    "accuracy for generalization. The sweet spot is somewhere in the middle -- the "
    "bias-variance trade-off in its purest form."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 -- Curse of Dimensionality
# ══════════════════════════════════════════════════════════════════════════════
st.header("4. The Curse of Dimensionality")

concept_box(
    "Why High Dimensions are Problematic for KNN",
    "Imagine you are playing darts at a dartboard. In 1D, points are close together on "
    "a line. In 2D, they spread across a plane. In 100D, they spread across a "
    "100-dimensional hyperspace, and here is the horrifying math: the volume of that "
    "space grows exponentially. With 4 features, we need a few thousand points to "
    "densely fill the space. With 100 features, we would need more points than atoms "
    "in the observable universe.<br><br>"
    "In practice, this means:<br>"
    "1. All points become roughly equidistant (so 'nearest neighbor' becomes meaningless)<br>"
    "2. The data becomes incredibly sparse<br>"
    "3. You need exponentially more data to maintain the same density<br><br>"
    "Our 4-feature dataset is well within safe territory, but if you ever try KNN on "
    "text data with 10,000 word features, you will see the curse in action."
)

# Demonstrate with subsets of features
st.subheader("Accuracy with Different Numbers of Features")
feature_subsets = {
    "1 feature (temp)": ["temperature_c"],
    "2 features (temp, humid)": ["temperature_c", "relative_humidity_pct"],
    "3 features (+wind)": ["temperature_c", "relative_humidity_pct", "wind_speed_kmh"],
    "4 features (all)": FEATURE_COLS,
}

subset_results = []
for name, feats in feature_subsets.items():
    X_tr_sub, X_te_sub, y_tr_sub, y_te_sub, _, _ = prepare_classification_data(
        filt, feats, target="city", test_size=0.2, scale=True, seed=42
    )
    knn_sub = KNeighborsClassifier(n_neighbors=5)
    knn_sub.fit(X_tr_sub, y_tr_sub)
    subset_results.append({
        "Features": name,
        "Num Features": len(feats),
        "Accuracy": knn_sub.score(X_te_sub, y_te_sub)
    })

subset_df = pd.DataFrame(subset_results)
fig_subset = px.bar(
    subset_df, x="Features", y="Accuracy",
    title="KNN Accuracy with Increasing Features",
    color="Accuracy", color_continuous_scale="Viridis",
    text_auto=".1%"
)
apply_common_layout(fig_subset, title="KNN Accuracy with Increasing Features", height=400)
st.plotly_chart(fig_subset, use_container_width=True)

insight_box(
    "With 4 features, adding more information helps. Each new feature gives KNN another "
    "axis along which to distinguish cities, and the data is dense enough for 'nearest "
    "neighbor' to still be meaningful. We are living in the happy regime where more "
    "features means more signal. But if we had 500 features, most of them noise, "
    "KNN would drown -- the curse of dimensionality would turn 'nearest neighbor' "
    "into 'random neighbor.'"
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 -- Distance Metrics Comparison
# ══════════════════════════════════════════════════════════════════════════════
st.header("5. Distance Metrics Comparison")

st.markdown(
    "You might wonder: does it matter which distance metric we use? The short answer "
    "is 'less than you think.' Let us check."
)

distance_metrics = ["euclidean", "manhattan", "chebyshev", "minkowski"]
metric_results = []

for metric in distance_metrics:
    kw = {"metric": metric}
    if metric == "minkowski":
        kw["p"] = 3  # Minkowski with p=3
    knn_met = KNeighborsClassifier(n_neighbors=K_val, **kw)
    knn_met.fit(X_train_s, y_train_s)
    metric_results.append({
        "Distance Metric": metric.title(),
        "Accuracy": knn_met.score(X_test_s, y_test_s),
    })

met_df = pd.DataFrame(metric_results)
fig_met = px.bar(
    met_df, x="Distance Metric", y="Accuracy",
    title=f"KNN Accuracy by Distance Metric (K={K_val})",
    color="Accuracy", color_continuous_scale="Blues",
    text_auto=".1%"
)
apply_common_layout(fig_met, title=f"KNN Accuracy by Distance Metric (K={K_val})", height=350)
st.plotly_chart(fig_met, use_container_width=True)

st.markdown(
    "As predicted: the differences are small. For most tabular data, Euclidean and "
    "Manhattan distances perform within a percentage point of each other. The choice "
    "of K and proper scaling matter about 10 times more than the distance metric. "
    "Do not spend hours agonizing over this."
)

code_example("""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# CRITICAL: Scale features before KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN with K=5
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_scaled, y_train)
print(f"Accuracy: {knn.score(X_test_scaled, y_test):.3f}")

# Find optimal K
from sklearn.model_selection import cross_val_score
for k in [1, 3, 5, 7, 11, 21]:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    print(f"K={k}: {scores.mean():.3f} +/- {scores.std():.3f}")
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 -- Quiz & Takeaways
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

quiz(
    "Why does KNN perform poorly on unscaled weather data?",
    [
        "KNN does not work with continuous features",
        "Surface pressure (~1000 hPa) dominates the distance calculation over temperature (~20 °C)",
        "KNN requires categorical features",
        "The dataset is too small for KNN",
    ],
    correct_idx=1,
    explanation="Without scaling, Euclidean distance is dominated by whichever feature has the "
    "largest numeric range. Pressure values are roughly 50x larger than temperature values, "
    "so the 'nearest neighbor' is really just the 'nearest pressure neighbor,' ignoring "
    "everything else.",
    key="q_knn_1"
)

quiz(
    "What happens when K = 1 in KNN?",
    [
        "The model always predicts the majority class",
        "Training accuracy is 100% but the model overfits",
        "The model ignores all features",
        "The model becomes a linear classifier",
    ],
    correct_idx=1,
    explanation="With K=1, each training point is trivially its own nearest neighbor, so training "
    "accuracy is always 100%. But this is pure memorization -- it learns the noise along "
    "with the signal. Set K=1 and you get a model with a photographic memory and zero judgment.",
    key="q_knn_2"
)

takeaways([
    "KNN is a lazy learner: no training phase, all computation happens at prediction time. The data IS the model.",
    "Feature scaling is not optional for KNN -- it is mandatory. Unscaled features let large-magnitude features dominate the distance calculation entirely.",
    "Small K = complex boundary (overfitting risk); large K = smooth boundary (underfitting risk). The bias-variance trade-off in its purest form.",
    "The curse of dimensionality means KNN struggles in very high dimensions, where all points become roughly equidistant and 'nearest' becomes meaningless.",
    "Unscaled weather data: pressure drowns out everything. Scaled data dramatically improves accuracy. This is the most important lesson of this chapter.",
    "KNN is simple to understand, simple to implement, and surprisingly hard to beat on small, well-scaled datasets. But it does not scale to large data well.",
])
