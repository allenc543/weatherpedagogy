"""Chapter 41 -- Feature Selection: Filter, Wrapper, and Embedded Methods."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.metrics import accuracy_score

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(41, "Feature Selection", part="IX")

st.markdown(
    "In the last chapter, we created a bunch of features. This chapter is about "
    "figuring out which of them actually matter. More features is not always better -- "
    "in fact, adding irrelevant features can actively hurt your model through the "
    "'curse of dimensionality' (high-dimensional spaces are weird and sparse), "
    "increased overfitting risk, and longer training times. Feature selection is the "
    "disciplined art of keeping the signal and throwing away the noise. There are "
    "three major philosophies, and they give surprisingly different answers."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

concept_box(
    "Three Approaches to Feature Selection",
    "<b>Filter methods</b>: Rank features by a statistical score (correlation, "
    "mutual information) without training any model. Fast and model-agnostic, "
    "but they cannot detect feature interactions.<br>"
    "<b>Wrapper methods</b>: Actually train models on different feature subsets "
    "and see which subset performs best (forward selection, RFE). More accurate "
    "but computationally expensive -- you are training many models.<br>"
    "<b>Embedded methods</b>: Feature selection happens *during* model training "
    "(Lasso regularization, tree-based importance). Best of both worlds in theory: "
    "model-aware but only requires one training run."
)

# ── Prepare extended feature set ─────────────────────────────────────────────
st.sidebar.subheader("Feature Selection Settings")
method = st.sidebar.selectbox(
    "Method",
    ["Mutual Information (Filter)", "Recursive Feature Elimination (Wrapper)",
     "Lasso Path (Embedded)", "Compare All"],
    key="fs_method",
)

# Build a richer feature set for more interesting selection
sub = fdf.dropna(subset=FEATURE_COLS).sample(n=min(8000, len(fdf)), random_state=42).copy()
sub["hour_sin"] = np.sin(2 * np.pi * sub["hour"] / 24)
sub["hour_cos"] = np.cos(2 * np.pi * sub["hour"] / 24)
sub["month_sin"] = np.sin(2 * np.pi * sub["month"] / 12)
sub["month_cos"] = np.cos(2 * np.pi * sub["month"] / 12)
sub["dew_point"] = sub["temperature_c"] - (100 - sub["relative_humidity_pct"]) / 5

all_features = FEATURE_COLS + [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dew_point",
]
feature_labels_ext = {
    **FEATURE_LABELS,
    "hour_sin": "Hour (sin)", "hour_cos": "Hour (cos)",
    "month_sin": "Month (sin)", "month_cos": "Month (cos)",
    "dew_point": "Dew Point (est.)",
}

le = LabelEncoder()
X = sub[all_features].values
y = le.fit_transform(sub["city"])
labels = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ── Section 1: Mutual Information (Filter) ───────────────────────────────────
if method in ["Mutual Information (Filter)", "Compare All"]:
    st.header("1. Mutual Information Scores (Filter Method)")

    st.markdown(
        "Mutual information (MI) asks a beautiful question: how much does knowing "
        "the value of feature X reduce my uncertainty about the target Y? Unlike "
        "Pearson correlation, MI captures **nonlinear** dependencies -- if temperature "
        "predicts city in a curvy, complicated way, MI will still pick it up. "
        "An MI score of 0 means the feature is completely independent of the target. "
        "Higher is better."
    )

    formula_box(
        "Mutual Information",
        r"\underbrace{I(X; Y)}_{\text{shared information}} = \sum_{x,y} \underbrace{p(x,y)}_{\text{joint probability}} \log \frac{\underbrace{p(x,y)}_{\text{joint probability}}}{\underbrace{p(x)}_{\text{feature prior}} \underbrace{p(y)}_{\text{target prior}}}",
        "MI = 0 when X and Y are independent; higher values mean stronger "
        "dependence. It is like correlation's cooler, more general cousin."
    )

    mi_scores = mutual_info_classif(X_train_s, y_train, random_state=42)
    mi_df = pd.DataFrame({
        "Feature": [feature_labels_ext.get(f, f) for f in all_features],
        "MI Score": mi_scores,
    }).sort_values("MI Score", ascending=True)

    fig_mi = go.Figure()
    fig_mi.add_trace(go.Bar(
        x=mi_df["MI Score"], y=mi_df["Feature"],
        orientation="h", marker_color="#2E86C1",
    ))
    fig_mi.update_layout(yaxis_title="", xaxis_title="Mutual Information Score")
    apply_common_layout(fig_mi, "Mutual Information Scores for City Classification", 500)
    st.plotly_chart(fig_mi, use_container_width=True)

    top3 = mi_df.nlargest(3, "MI Score")["Feature"].tolist()
    insight_box(
        f"The top 3 most informative features are: **{', '.join(top3)}**. "
        "These carry the most information for distinguishing between cities. "
        "Notice that MI can rank features very differently from simple correlation -- "
        "it captures all the nonlinear structure too."
    )

# ── Section 2: Recursive Feature Elimination (Wrapper) ──────────────────────
if method in ["Recursive Feature Elimination (Wrapper)", "Compare All"]:
    st.header("2. Recursive Feature Elimination (RFE)")

    st.markdown(
        "RFE is the brute-force approach, and sometimes brute force is exactly right. "
        "It starts with all features, trains a model, identifies the least important "
        "feature (based on the model's own importance scores), removes it, and repeats. "
        "By the end, you have a ranking from most to least important, determined by "
        "how much removing each feature actually hurts the model."
    )

    n_select = st.slider(
        "Number of features to select",
        1, len(all_features), 5, key="rfe_n",
    )

    with st.spinner("Running RFE..."):
        estimator = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        rfe = RFE(estimator, n_features_to_select=n_select, step=1)
        rfe.fit(X_train_s, y_train)

    rfe_df = pd.DataFrame({
        "Feature": [feature_labels_ext.get(f, f) for f in all_features],
        "Ranking": rfe.ranking_,
        "Selected": rfe.support_,
    }).sort_values("Ranking")

    st.markdown(f"**Features selected ({n_select}):**")
    selected_feats = rfe_df[rfe_df["Selected"]]["Feature"].tolist()
    for f in selected_feats:
        st.markdown(f"- {f}")

    st.dataframe(rfe_df, use_container_width=True, hide_index=True)

    # Performance with selected vs all features
    acc_all = accuracy_score(y_test, estimator.fit(X_train_s, y_train).predict(X_test_s))
    X_train_rfe = X_train_s[:, rfe.support_]
    X_test_rfe = X_test_s[:, rfe.support_]
    acc_rfe = accuracy_score(y_test, estimator.fit(X_train_rfe, y_train).predict(X_test_rfe))

    c1, c2 = st.columns(2)
    c1.metric("Accuracy (all features)", f"{acc_all:.1%}")
    c2.metric(f"Accuracy ({n_select} features)", f"{acc_rfe:.1%}")

    insight_box(
        f"RFE selected {n_select} out of {len(all_features)} features and achieves "
        f"**{acc_rfe:.1%}** accuracy vs **{acc_all:.1%}** with all features. "
        "If the accuracy barely changes, then the dropped features were dead weight. "
        "A simpler model with similar accuracy is almost always preferable -- it is "
        "faster, more interpretable, and less prone to overfitting."
    )

# ── Section 3: Lasso Path (Embedded) ────────────────────────────────────────
if method in ["Lasso Path (Embedded)", "Compare All"]:
    st.header("3. Lasso Regularization Path (Embedded Method)")

    st.markdown(
        "Lasso is the sniper rifle of feature selection. It adds an L1 penalty "
        "to the loss function that literally drives unimportant feature coefficients "
        "to exactly zero. Not 'close to zero' -- *exactly* zero. This means Lasso "
        "simultaneously fits the model and performs feature selection, which is "
        "elegant and computationally efficient. The regularization path below "
        "shows how features enter and exit the model as we dial up the penalty."
    )

    formula_box(
        "Lasso Objective",
        r"\min_\beta \underbrace{\frac{1}{2n} \|y - X\beta\|^2_2}_{\text{prediction error}} + \underbrace{\alpha}_{\text{penalty strength}} \underbrace{\|\beta\|_1}_{\text{sum of abs coefficients}}",
        "The L1 penalty (alpha times the sum of absolute coefficient values) "
        "encourages sparsity. Bigger alpha = more features driven to zero."
    )

    from sklearn.linear_model import LogisticRegression as LR

    # Compute Lasso path for different C values (C = 1/alpha)
    C_values = np.logspace(-3, 2, 30)
    coef_paths = {f: [] for f in all_features}
    for C in C_values:
        lr = LR(penalty="l1", C=C, solver="saga", max_iter=2000,
                multi_class="multinomial", random_state=42)
        lr.fit(X_train_s, y_train)
        avg_coef = np.mean(np.abs(lr.coef_), axis=0)
        for i, f in enumerate(all_features):
            coef_paths[f].append(avg_coef[i])

    fig_path = go.Figure()
    colors_path = px.colors.qualitative.Set2 + px.colors.qualitative.Set1
    for i, f in enumerate(all_features):
        fig_path.add_trace(go.Scatter(
            x=np.log10(C_values), y=coef_paths[f],
            mode="lines", name=feature_labels_ext.get(f, f),
            line=dict(color=colors_path[i % len(colors_path)], width=2),
        ))
    fig_path.update_layout(
        xaxis_title="log10(C)  [C = 1/alpha; left = more regularization]",
        yaxis_title="Mean |Coefficient|",
    )
    apply_common_layout(fig_path, "Lasso Regularization Path", 500)
    st.plotly_chart(fig_path, use_container_width=True)

    insight_box(
        "Features that remain non-zero under strong regularization (left side "
        "of the plot) are the ones Lasso considers truly essential. They survive "
        "even when the penalty is cranked up. Features that quickly drop to zero "
        "are the dispensable ones -- the model can do without them."
    )

# ── Section 4: Forward Selection Process ─────────────────────────────────────
if method in ["Compare All"]:
    st.header("4. Forward Selection (Greedy Wrapper)")

    st.markdown(
        "Forward selection is the opposite of RFE: it starts with *no* features "
        "and adds them one at a time, always picking whichever single feature "
        "improves accuracy the most at each step. It is greedy (it never "
        "reconsiders previous choices), but it gives you a nice, interpretable "
        "trajectory of 'most to least important feature to add.'"
    )

    selected = []
    remaining = list(range(len(all_features)))
    forward_log = []

    base_model = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1)

    for step in range(min(7, len(all_features))):
        best_acc = 0
        best_feat = None
        for feat_idx in remaining:
            trial = selected + [feat_idx]
            base_model.fit(X_train_s[:, trial], y_train)
            acc = accuracy_score(y_test, base_model.predict(X_test_s[:, trial]))
            if acc > best_acc:
                best_acc = acc
                best_feat = feat_idx
        selected.append(best_feat)
        remaining.remove(best_feat)
        forward_log.append({
            "Step": step + 1,
            "Feature Added": feature_labels_ext.get(all_features[best_feat], all_features[best_feat]),
            "Accuracy": round(best_acc, 4),
        })

    fwd_df = pd.DataFrame(forward_log)
    st.dataframe(fwd_df, use_container_width=True, hide_index=True)

    fig_fwd = go.Figure()
    fig_fwd.add_trace(go.Scatter(
        x=fwd_df["Step"], y=fwd_df["Accuracy"],
        mode="lines+markers+text",
        text=fwd_df["Feature Added"],
        textposition="top center",
        marker=dict(size=10, color="#E63946"),
        line=dict(color="#E63946", width=2),
    ))
    fig_fwd.update_layout(xaxis_title="Number of Features", yaxis_title="Accuracy")
    apply_common_layout(fig_fwd, "Forward Selection: Accuracy vs Number of Features", 450)
    st.plotly_chart(fig_fwd, use_container_width=True)

# ── Section 5: Comparison of Feature Subsets ─────────────────────────────────
st.header("5. Model Performance by Feature Subset Size")

st.markdown(
    "Here is the key experiment. We rank features by mutual information, then "
    "train a Random Forest using the top-1 feature, top-2 features, top-3, and "
    "so on. The resulting accuracy curve tells you where the diminishing returns "
    "kick in -- the point where adding more features stops helping and starts "
    "just adding noise and computation."
)

mi_scores_all = mutual_info_classif(X_train_s, y_train, random_state=42)
mi_order = np.argsort(mi_scores_all)[::-1]

subset_results = []
rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
for k in range(1, len(all_features) + 1):
    top_k = mi_order[:k]
    rf.fit(X_train_s[:, top_k], y_train)
    acc = accuracy_score(y_test, rf.predict(X_test_s[:, top_k]))
    subset_results.append({"Num Features": k, "Accuracy": round(acc, 4)})

sub_df = pd.DataFrame(subset_results)
fig_sub = go.Figure()
fig_sub.add_trace(go.Scatter(
    x=sub_df["Num Features"], y=sub_df["Accuracy"],
    mode="lines+markers",
    marker=dict(size=8, color="#2E86C1"),
    line=dict(color="#2E86C1", width=2),
))
fig_sub.update_layout(
    xaxis_title="Number of Features (ranked by MI)",
    yaxis_title="Accuracy",
    xaxis=dict(dtick=1),
)
apply_common_layout(fig_sub, "Accuracy vs Number of Features", 400)
st.plotly_chart(fig_sub, use_container_width=True)

# Find elbow
diffs = np.diff([r["Accuracy"] for r in subset_results])
elbow = np.argmin(np.abs(diffs)) + 1  # where adding more features stops helping
insight_box(
    f"Performance plateaus around **{elbow}-{elbow+1} features**. Beyond that "
    "point, each additional feature contributes less than the last, and you are "
    "essentially paying in complexity for pocket change in accuracy. This is the "
    "feature selection sweet spot."
)

warning_box(
    "In a real-world workflow, feature selection should be done with cross-validation, "
    "not a single train/test split. The particular split can make a feature look "
    "more or less important than it really is. We use a single split here for "
    "speed and clarity, but do not do this in production."
)

code_example("""
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# Filter: Mutual Information
mi_scores = mutual_info_classif(X_train, y_train)

# Wrapper: RFE
rfe = RFE(RandomForestClassifier(n_estimators=50), n_features_to_select=5)
rfe.fit(X_train, y_train)
selected = rfe.support_

# Embedded: Lasso
from sklearn.linear_model import LogisticRegression
lasso = LogisticRegression(penalty='l1', C=0.1, solver='saga')
lasso.fit(X_train, y_train)
important = np.where(np.any(lasso.coef_ != 0, axis=0))[0]
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "Which feature selection approach uses the model's own training process "
    "to eliminate unimportant features?",
    [
        "Filter method (e.g., mutual information)",
        "Wrapper method (e.g., forward selection)",
        "Embedded method (e.g., Lasso regularization)",
        "Random selection",
    ],
    2,
    "Embedded methods perform feature selection *during* model training. Lasso "
    "literally drives unimportant coefficients to zero as part of the optimization "
    "process. It is feature selection and model fitting in one step.",
    key="fs_quiz",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "Feature selection reduces overfitting, speeds up training, and improves interpretability. More features is not always better.",
    "**Filter** methods (MI, correlation) are fast and model-agnostic, but cannot detect feature interactions.",
    "**Wrapper** methods (RFE, forward selection) use actual model accuracy as the criterion, but are computationally expensive because they train many models.",
    "**Embedded** methods (Lasso, tree importance) perform selection during training -- elegant and efficient.",
    "Performance often plateaus after a handful of features -- the accuracy curve typically has an elbow, and adding features past it gives diminishing returns.",
    "Always validate feature selection with cross-validation. A single split can be misleading.",
])
