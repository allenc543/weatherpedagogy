"""Chapter 24: Support Vector Machines -- Margins, kernels, and decision boundaries."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.ml_helpers import prepare_classification_data, classification_metrics, plot_confusion_matrix
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(24, "Support Vector Machines", part="V")
st.markdown(
    "Most classifiers draw a boundary between classes and call it a day. SVMs are "
    "pickier: they insist on finding the boundary that is as **far as possible from "
    "both sides**. Imagine two crowds separated by a fence -- the SVM does not just "
    "build any fence, it builds the fence that maximizes the no-man's-land on either "
    "side. And with kernel tricks, it can build curvy fences in dimensions your eyes "
    "cannot see. It is mathematically beautiful and surprisingly effective, though it "
    "has some scaling issues we should talk about."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
filt = sidebar_filters(df)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 -- Margins and Support Vectors
# ══════════════════════════════════════════════════════════════════════════════
st.header("1. Margins and Support Vectors")

concept_box(
    "Maximum Margin Classifier",
    "Among all possible separating hyperplanes, the SVM chooses the one with the "
    "<b>largest margin</b> -- the widest possible gap between classes. Why does this "
    "matter? Because a boundary with a wide margin is more robust to new data. If your "
    "fence is right up against one crowd, any small movement could put someone on the "
    "wrong side. A wide margin gives you breathing room.<br><br>"
    "The data points sitting right on the edge of the margin are called <b>support "
    "vectors</b>. They are the only points that actually matter -- you could throw "
    "away every other training example and the boundary would not change."
)

formula_box(
    "SVM Optimization Objective",
    r"\min_{\underbrace{\mathbf{w}, b}_{\text{boundary params}}} \underbrace{\frac{1}{2}\|\mathbf{w}\|^2}_{\text{maximize margin}} \quad \text{s.t. } \underbrace{y_i}_{\text{true label}} \underbrace{(\mathbf{w}^\top \mathbf{x}_i + b)}_{\text{decision score}} \geq 1, \; \forall i",
    "Minimize the norm of the weight vector (which is equivalent to maximizing the margin) "
    "subject to all points being on the correct side. The math is elegant: a constrained "
    "optimization problem with a quadratic objective and linear constraints."
)

# Simple 2D illustration
st.subheader("Intuition: Maximizing the Margin")
st.markdown(
    "Imagine separating NYC weather (cold, humid winters) from LA (mild, dry). There "
    "are infinitely many lines you could draw between them. A logistic regression would "
    "pick whichever line minimizes misclassification. The SVM is more demanding: it "
    "picks the line that is **farthest from both sides**. It is optimizing for worst-case "
    "robustness, which is a fundamentally different philosophy."
)

# Interactive illustration with synthetic data
np.random.seed(42)
n_pts = 50
class_a = np.random.randn(n_pts, 2) * 0.8 + np.array([-2, -1])
class_b = np.random.randn(n_pts, 2) * 0.8 + np.array([2, 1])
demo_X = np.vstack([class_a, class_b])
demo_y = np.array([0]*n_pts + [1]*n_pts)

svm_demo = SVC(kernel="linear", C=1.0)
svm_demo.fit(demo_X, demo_y)
w = svm_demo.coef_[0]
b_svm = svm_demo.intercept_[0]
sv_idx = svm_demo.support_

x_line = np.linspace(-5, 5, 100)
y_line = -(w[0] * x_line + b_svm) / w[1]
y_margin_up = -(w[0] * x_line + b_svm - 1) / w[1]
y_margin_down = -(w[0] * x_line + b_svm + 1) / w[1]

fig_margin = go.Figure()
fig_margin.add_trace(go.Scatter(x=class_a[:, 0], y=class_a[:, 1], mode="markers",
                                 name="Class A", marker=dict(color="#7209B7", size=8)))
fig_margin.add_trace(go.Scatter(x=class_b[:, 0], y=class_b[:, 1], mode="markers",
                                 name="Class B", marker=dict(color="#FB8500", size=8)))
# Support vectors
fig_margin.add_trace(go.Scatter(
    x=demo_X[sv_idx, 0], y=demo_X[sv_idx, 1], mode="markers",
    name="Support Vectors", marker=dict(color="red", size=14, symbol="circle-open", line=dict(width=3))
))
# Decision boundary and margins
fig_margin.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Decision Boundary",
                                 line=dict(color="black", width=2)))
fig_margin.add_trace(go.Scatter(x=x_line, y=y_margin_up, mode="lines", name="Margin",
                                 line=dict(color="gray", width=1, dash="dash")))
fig_margin.add_trace(go.Scatter(x=x_line, y=y_margin_down, mode="lines", showlegend=False,
                                 line=dict(color="gray", width=1, dash="dash")))

apply_common_layout(fig_margin, title="SVM: Maximum Margin with Support Vectors", height=450)
fig_margin.update_layout(xaxis_range=[-5, 5], yaxis_range=[-4, 4])
st.plotly_chart(fig_margin, use_container_width=True)

insight_box(
    f"Only **{len(sv_idx)} support vectors** (circled in red) out of {len(demo_X)} total "
    "points determine the entire decision boundary. You could delete, move, or replace "
    "every other point and the boundary would not budge. The SVM has decided that "
    f"about {len(sv_idx)/len(demo_X)*100:.0f}% of the data is the only part that "
    "actually matters."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 -- Kernel Trick
# ══════════════════════════════════════════════════════════════════════════════
st.header("2. The Kernel Trick")

concept_box(
    "When Data is Not Linearly Separable",
    "Here is the SVM's party trick: the <b>kernel trick</b>. When data cannot be "
    "separated by a straight line in the original space, you map it into a "
    "higher-dimensional space where a linear boundary exists. The genius part is that "
    "you never actually compute the high-dimensional coordinates -- you just compute "
    "distances using a kernel function, which is dramatically cheaper.<br><br>"
    "Common kernels:<br>"
    "- <b>Linear</b>: K(x,x') = x^T x' (no transformation, just a straight line)<br>"
    "- <b>RBF</b>: K(x,x') = exp(-gamma ||x-x'||^2) (implicitly maps to infinite dimensions!)<br>"
    "- <b>Polynomial</b>: K(x,x') = (gamma x^T x' + r)^d (maps to a finite but high-dimensional space)"
)

formula_box(
    "RBF Kernel",
    r"\underbrace{K(\mathbf{x}, \mathbf{x}')}_{\text{kernel similarity}} = \exp\!\left(-\underbrace{\gamma}_{\text{reach parameter}} \underbrace{\|\mathbf{x} - \mathbf{x}'\|^2}_{\text{squared distance}}\right)",
    "Gamma controls how far the influence of a single training point reaches. "
    "Small gamma = far reach, smooth boundary. Large gamma = close reach, wiggly boundary "
    "that hugs each data point. Think of gamma as the SVM's magnifying glass: high gamma "
    "means it is zooming in very close."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 -- Interactive NYC vs LA with Decision Boundary
# ══════════════════════════════════════════════════════════════════════════════
st.header("3. Interactive: NYC vs LA Decision Boundary")

st.markdown(
    "Here you can pick a kernel, tune the hyperparameters, and watch the decision "
    "boundary change in real time. We use 2 features so we can actually visualize "
    "what is happening. Try switching between linear and RBF -- the difference is striking."
)

col_s1, col_s2 = st.columns(2)
with col_s1:
    feat_x = st.selectbox("X-axis feature", FEATURE_COLS, index=0, key="svm_fx")
with col_s2:
    feat_y = st.selectbox("Y-axis feature", FEATURE_COLS, index=3, key="svm_fy")

kernel_choice = st.selectbox("Kernel", ["linear", "rbf", "poly"], index=1, key="svm_kernel")
C_val = st.slider("C (regularization)", 0.01, 10.0, 1.0, 0.1, key="svm_c")

gamma_options = {"auto": "auto"}
if kernel_choice in ("rbf", "poly"):
    gamma_val = st.slider("Gamma", 0.001, 2.0, 0.1, 0.01, key="svm_gamma")
else:
    gamma_val = "auto"

if kernel_choice == "poly":
    degree_val = st.slider("Polynomial degree", 2, 5, 3, 1, key="svm_degree")
else:
    degree_val = 3

# Prepare binary data
df_binary = df[df["city"].isin(["NYC", "Los Angeles"])].copy()
le_bin = LabelEncoder()
X_2d_raw = df_binary[[feat_x, feat_y]].dropna()
y_2d = le_bin.fit_transform(df_binary.loc[X_2d_raw.index, "city"])

# Scale for SVM
scaler_2d = StandardScaler()
X_2d_scaled = scaler_2d.fit_transform(X_2d_raw)
X_2d_df = pd.DataFrame(X_2d_scaled, columns=[feat_x, feat_y], index=X_2d_raw.index)

# Subsample for speed
np.random.seed(42)
if len(X_2d_df) > 5000:
    idx = np.random.choice(len(X_2d_df), 5000, replace=False)
    X_sub = X_2d_df.iloc[idx]
    y_sub = y_2d[idx]
else:
    X_sub = X_2d_df
    y_sub = y_2d

svm_model = SVC(kernel=kernel_choice, C=C_val, gamma=gamma_val, degree=degree_val, random_state=42)
svm_model.fit(X_sub, y_sub)
svm_acc = svm_model.score(X_sub, y_sub)

# Decision boundary mesh
h = 0.05
x_min, x_max = X_sub[feat_x].min() - 0.5, X_sub[feat_x].max() + 0.5
y_min, y_max = X_sub[feat_y].min() - 0.5, X_sub[feat_y].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig_svm = go.Figure()
fig_svm.add_trace(go.Contour(
    x=np.arange(x_min, x_max, h), y=np.arange(y_min, y_max, h),
    z=Z, colorscale=[[0, "#D8BFD8"], [1, "#FFE4B5"]], opacity=0.5,
    showscale=False, contours=dict(showlines=True, coloring="fill")
))

labels_bin = le_bin.classes_.tolist()
for cls_idx, city in enumerate(labels_bin):
    mask = y_sub == cls_idx
    # Show a subsample in the plot
    plot_idx = np.where(mask)[0]
    if len(plot_idx) > 500:
        plot_idx = np.random.choice(plot_idx, 500, replace=False)
    fig_svm.add_trace(go.Scatter(
        x=X_sub.iloc[plot_idx][feat_x], y=X_sub.iloc[plot_idx][feat_y],
        mode="markers", name=city, opacity=0.5,
        marker=dict(color=CITY_COLORS.get(city, "gray"), size=4)
    ))

# Highlight support vectors
sv = svm_model.support_
if len(sv) > 200:
    sv_show = np.random.choice(sv, 200, replace=False)
else:
    sv_show = sv
fig_svm.add_trace(go.Scatter(
    x=X_sub.iloc[sv_show][feat_x], y=X_sub.iloc[sv_show][feat_y],
    mode="markers", name="Support Vectors",
    marker=dict(color="red", size=6, symbol="circle-open", line=dict(width=1.5))
))

apply_common_layout(fig_svm, title=f"SVM Decision Boundary ({kernel_choice} kernel)", height=500)
fig_svm.update_layout(
    xaxis_title=f"{FEATURE_LABELS.get(feat_x, feat_x)} (scaled)",
    yaxis_title=f"{FEATURE_LABELS.get(feat_y, feat_y)} (scaled)",
)
st.plotly_chart(fig_svm, use_container_width=True)

col_a, col_b = st.columns(2)
col_a.metric("Training Accuracy (2D)", f"{svm_acc:.1%}")
col_b.metric("Support Vectors", len(svm_model.support_))

if kernel_choice == "linear":
    insight_box(
        "With a linear kernel, the decision boundary is a straight line. Refreshingly "
        "simple and often good enough when NYC and LA are well-separated along the chosen "
        "features. Sometimes the right answer really is just a line."
    )
elif kernel_choice == "rbf":
    insight_box(
        "The RBF kernel creates a flexible, non-linear boundary that curves around the "
        "data. Crank gamma up and watch the boundary become increasingly wiggly, trying "
        "to wrap around each individual point. That is overfitting in real time -- the "
        "SVM is memorizing rather than generalizing."
    )
else:
    insight_box(
        "The polynomial kernel creates curved boundaries whose complexity is controlled "
        "by the degree parameter. Degree 2 gives you ellipses and parabolas; degree 5 "
        "gives you boundaries that look like they were drawn by a caffeinated octopus."
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 -- C Parameter Effect
# ══════════════════════════════════════════════════════════════════════════════
st.header("4. The C Parameter: Hard vs Soft Margins")

concept_box(
    "Regularization with C",
    "Real data is messy. Points from different classes overlap. A hard margin -- insisting "
    "on zero misclassifications -- might not even be possible, and if it is, the boundary "
    "might be razor-thin and fragile. C controls the trade-off:<br>"
    "- <b>Large C</b>: 'I really care about getting training points right.' Narrow margin, "
    "fewer misclassifications, may overfit.<br>"
    "- <b>Small C</b>: 'I am okay with a few mistakes if it means a wider, more robust "
    "margin.' May underfit on training data but generalize better."
)

c_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
c_accs = []
c_svs = []
for c in c_values:
    svm_c = SVC(kernel=kernel_choice, C=c, gamma=gamma_val, degree=degree_val, random_state=42)
    svm_c.fit(X_sub, y_sub)
    c_accs.append(svm_c.score(X_sub, y_sub))
    c_svs.append(len(svm_c.support_))

c_df = pd.DataFrame({"C": c_values, "Accuracy": c_accs, "Support Vectors": c_svs})

fig_c = go.Figure()
fig_c.add_trace(go.Scatter(
    x=c_df["C"], y=c_df["Accuracy"], mode="lines+markers",
    name="Accuracy", yaxis="y1", line=dict(color="#2E86C1", width=3)
))
fig_c.add_trace(go.Scatter(
    x=c_df["C"], y=c_df["Support Vectors"], mode="lines+markers",
    name="Support Vectors", yaxis="y2", line=dict(color="#E63946", width=3)
))
fig_c.update_layout(
    title="Effect of C on Accuracy and Number of Support Vectors",
    xaxis=dict(title="C (regularization)", type="log"),
    yaxis=dict(title="Accuracy", side="left", titlefont=dict(color="#2E86C1")),
    yaxis2=dict(title="Support Vectors", side="right", overlaying="y",
                titlefont=dict(color="#E63946")),
    template="plotly_white", height=400,
)
st.plotly_chart(fig_c, use_container_width=True)

insight_box(
    "As C increases, the SVM becomes more obsessive about classifying training points "
    "correctly, which pushes accuracy up -- but also reduces the number of support "
    "vectors. Fewer support vectors means the boundary depends on fewer points, which "
    "sounds good until you realize it might be depending on noise. Very high C is the "
    "SVM equivalent of 'trust me bro, I have memorized everything.'"
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 -- Full Multi-class SVM
# ══════════════════════════════════════════════════════════════════════════════
st.header("5. Full 6-City SVM Classification")

st.markdown(
    "Enough with the 2D binary case. Let us unleash the SVM on all 6 cities with all "
    "4 features and see what happens."
)

X_train_m, X_test_m, y_train_m, y_test_m, le_m, scaler_m = prepare_classification_data(
    filt, FEATURE_COLS, target="city", test_size=0.2, scale=True, seed=42
)

svm_full = SVC(kernel=kernel_choice, C=C_val, gamma=gamma_val, degree=degree_val, random_state=42)
svm_full.fit(X_train_m, y_train_m)
y_pred_full = svm_full.predict(X_test_m)
labels_full = le_m.classes_.tolist()
metrics_full = classification_metrics(y_test_m, y_pred_full, labels=labels_full)

col1, col2 = st.columns(2)
col1.metric("6-City Test Accuracy", f"{metrics_full['accuracy']:.1%}")
col2.metric("Support Vectors", len(svm_full.support_))

st.plotly_chart(
    plot_confusion_matrix(metrics_full["confusion_matrix"], labels_full),
    use_container_width=True
)

warning_box(
    "SVMs have a dirty secret: they do not scale well. Training time is roughly O(n^2) "
    "to O(n^3), which means doubling your data quadruples (or octuples!) your training "
    "time. Our 100K+ rows require subsampling for the 2D visualization. For large-scale "
    "production work, consider LinearSVC or SGDClassifier, which use clever approximations "
    "to avoid the full quadratic cost."
)

code_example("""
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# IMPORTANT: Always scale features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different kernels
for kernel in ['linear', 'rbf', 'poly']:
    svm = SVC(kernel=kernel, C=1.0, gamma='auto', random_state=42)
    svm.fit(X_train_scaled, y_train)
    print(f"{kernel}: {svm.score(X_test_scaled, y_test):.3f}")

# Access support vectors
print(f"Number of support vectors: {len(svm.support_)}")
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 -- Quiz & Takeaways
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

quiz(
    "What are support vectors?",
    [
        "All training data points",
        "The data points closest to the decision boundary that define the margin",
        "The centroids of each class",
        "Randomly selected data points",
    ],
    correct_idx=1,
    explanation="Support vectors are the critical points sitting right on the margin's edge. "
    "They are the only points the SVM actually cares about -- every other point could vanish "
    "and the boundary would stay the same.",
    key="q_svm_1"
)

quiz(
    "What does a larger C value do in an SVM?",
    [
        "Increases the margin width (more regularization)",
        "Decreases the margin width to classify more training points correctly",
        "Changes the kernel type",
        "Reduces the number of features",
    ],
    correct_idx=1,
    explanation="Larger C means 'I care a lot about getting every training point right,' which "
    "forces a narrower margin to accommodate more points on the correct side. Think of it as "
    "the SVM's perfectionism dial.",
    key="q_svm_2"
)

takeaways([
    "SVMs find the hyperplane that maximizes the margin between classes -- they optimize for worst-case robustness, not average performance.",
    "Support vectors are the data points on the margin's edge -- they alone define the boundary. Everything else is irrelevant.",
    "The kernel trick allows SVMs to learn non-linear boundaries by implicitly mapping data to higher-dimensional spaces.",
    "C controls the perfectionism: large C = narrow margin, fewer mistakes on training data, overfitting risk. Small C = wide margin, more tolerance, better generalization.",
    "Gamma (for RBF kernel) controls the reach of each support vector: high gamma = wiggly boundary, low gamma = smooth boundary.",
    "Feature scaling is absolutely critical for SVMs -- they measure distances, and unscaled features make distances meaningless.",
])
