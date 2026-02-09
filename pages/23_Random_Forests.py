"""Chapter 23: Random Forests -- Bootstrap aggregation and ensemble power."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.ml_helpers import prepare_classification_data, classification_metrics, plot_confusion_matrix
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(23, "Random Forests", part="V")
st.markdown(
    "In the last chapter, we saw that individual decision trees have a serious problem: "
    "they are brilliant at memorizing training data and mediocre at generalizing to new "
    "data. Random forests solve this with an almost comically simple idea: **train a "
    "bunch of mediocre trees that each see different data and different features, then "
    "let them vote**. It turns out that a committee of dumb-but-diverse models "
    "consistently outperforms any single smart model. There is a lesson about democracy "
    "in here somewhere."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
filt = sidebar_filters(df)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 -- Bootstrap Aggregation (Bagging)
# ══════════════════════════════════════════════════════════════════════════════
st.header("1. Bootstrap Aggregation (Bagging)")

concept_box(
    "The Core Idea",
    "Train many decision trees, but give each one a <b>different random subset</b> of "
    "the data (bootstrap samples -- random draws with replacement) and only let each "
    "split consider a <b>random subset of features</b>. Then let all the trees <b>vote</b> "
    "on the prediction. This reduces variance and makes the model far more robust "
    "than any single tree.<br><br>"
    "Why does this work? Because each tree overfits in a <i>different</i> way. Their "
    "errors are uncorrelated, so averaging them out cancels the noise. It is the same "
    "reason why asking 100 people to guess the number of jellybeans in a jar gives "
    "a better answer than asking 1 expert."
)

st.markdown("""
**How a Random Forest works:**
1. Draw N bootstrap samples (random sampling with replacement) from the training data
2. For each sample, train a decision tree using a random subset of features at each split
3. To predict: each tree votes, and the majority class wins

The randomness comes from two sources:
- **Bootstrap sampling** -- each tree sees different data
- **Feature randomization** -- each split considers only a random subset of features
""")

formula_box(
    "Ensemble Prediction (Majority Vote)",
    r"\hat{y} = \text{mode}\left(\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_B\right)",
    "Where B is the number of trees. Each tree casts one vote. The most popular answer "
    "wins. No weighting, no fancy aggregation -- just raw democracy."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 -- Single Tree vs Random Forest
# ══════════════════════════════════════════════════════════════════════════════
st.header("2. Single Decision Tree vs Random Forest")

X_train, X_test, y_train, y_test, le, scaler = prepare_classification_data(
    filt, FEATURE_COLS, target="city", test_size=0.2, scale=False, seed=42
)
labels = le.classes_.tolist()

# Single decision tree baseline
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_train_acc = dt_model.score(X_train, y_train)
dt_test_acc = dt_model.score(X_test, y_test)

# Random forest
n_estimators = st.slider("Number of trees (n_estimators)", 1, 200, 50, 1, key="rf_n_est")
max_depth_rf = st.slider("Max depth per tree", 3, 20, 10, 1, key="rf_depth")
max_features = st.selectbox(
    "Max features per split",
    ["sqrt", "log2", "all"],
    index=0,
    key="rf_max_feat"
)
max_feat_val = None if max_features == "all" else max_features

rf_model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth_rf,
    max_features=max_feat_val,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    oob_score=True if n_estimators > 1 else False,
)
rf_model.fit(X_train, y_train)
rf_train_acc = rf_model.score(X_train, y_train)
rf_test_acc = rf_model.score(X_test, y_test)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Single Decision Tree")
    st.metric("Train Accuracy", f"{dt_train_acc:.1%}")
    st.metric("Test Accuracy", f"{dt_test_acc:.1%}")
    st.metric("Overfitting Gap", f"{dt_train_acc - dt_test_acc:.1%}")

with col2:
    st.subheader("Random Forest")
    st.metric("Train Accuracy", f"{rf_train_acc:.1%}")
    st.metric("Test Accuracy", f"{rf_test_acc:.1%}")
    st.metric("Overfitting Gap", f"{rf_train_acc - rf_test_acc:.1%}")
    if n_estimators > 1 and hasattr(rf_model, 'oob_score_'):
        st.metric("OOB Score", f"{rf_model.oob_score_:.1%}")

improvement = rf_test_acc - dt_test_acc
if improvement > 0:
    insight_box(
        f"The random forest improves test accuracy by **{improvement:.1%}** over "
        "a single tree. Notice something else: the overfitting gap shrinks. The single "
        "tree memorizes training data; the forest averages out each tree's idiosyncratic "
        "mistakes. Diverse mediocrity beats brilliant fragility."
    )
else:
    st.info(
        "The random forest and single tree perform similarly here. Try increasing "
        "n_estimators or adjusting hyperparameters -- the forest needs enough trees "
        "for the averaging to kick in."
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 -- Watching Accuracy Improve with More Trees
# ══════════════════════════════════════════════════════════════════════════════
st.header("3. How Accuracy Improves with More Trees")

concept_box(
    "Out-of-Bag (OOB) Error",
    "Here is a charming bonus of bootstrap sampling: each tree only sees about 63% of "
    "the training data (the other 37% is 'out of bag'). You can use those left-out "
    "samples to estimate generalization error <b>for free, without a separate validation "
    "set</b>. It is like getting your practice exam graded by the questions you did not "
    "study."
)

st.markdown(
    "Watch the chart below as we add more trees. The accuracy does something wonderful: "
    "it improves rapidly, then plateaus. And here is the key insight -- it never really "
    "*degrades* with more trees. You cannot overfit by adding more trees to a random "
    "forest, which is a remarkably forgiving property."
)

n_tree_range = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200]
tree_accs = []

for n in n_tree_range:
    rf_temp = RandomForestClassifier(
        n_estimators=n, max_depth=max_depth_rf, max_features=max_feat_val,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf_temp.fit(X_train, y_train)
    tree_accs.append(rf_temp.score(X_test, y_test))

acc_df = pd.DataFrame({"Number of Trees": n_tree_range, "Test Accuracy": tree_accs})

fig_trees = px.line(
    acc_df, x="Number of Trees", y="Test Accuracy",
    title="Test Accuracy vs Number of Trees",
    markers=True,
)
fig_trees.update_traces(line=dict(color="#2E86C1", width=3))
apply_common_layout(fig_trees, title="Test Accuracy vs Number of Trees", height=400)
fig_trees.update_yaxes(range=[min(tree_accs) - 0.02, max(tree_accs) + 0.02])
st.plotly_chart(fig_trees, use_container_width=True)

insight_box(
    "The jump from 1 tree to 10 trees is dramatic. From 50 to 200? Barely noticeable. "
    "This is diminishing returns in action. The first few trees correct each other's "
    "worst mistakes; the later trees are just confirming what the committee already "
    "knows. In practice, 100-200 trees is almost always enough."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 -- Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
st.header("4. Feature Importance")

concept_box(
    "Random Forest Feature Importance",
    "A single decision tree's feature importance is brittle -- it depends on the exact "
    "training data and the order of splits. Random forests average importance across all "
    "trees, and because each tree uses a random subset of features, every feature gets "
    "its chance to prove itself. The result is a more stable, more trustworthy picture "
    "of what actually matters."
)

# Side-by-side comparison
rf_imp = rf_model.feature_importances_
dt_imp = dt_model.feature_importances_

imp_compare = pd.DataFrame({
    "Feature": [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
    "Single Tree": dt_imp,
    "Random Forest": rf_imp,
})
imp_melt = imp_compare.melt(id_vars="Feature", var_name="Model", value_name="Importance")

fig_imp = px.bar(
    imp_melt, x="Importance", y="Feature", color="Model",
    orientation="h", barmode="group",
    title="Feature Importance: Single Tree vs Random Forest",
    color_discrete_map={"Single Tree": "#E63946", "Random Forest": "#2E86C1"},
)
apply_common_layout(fig_imp, title="Feature Importance: Single Tree vs Random Forest", height=400)
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown(
    "Notice the pattern: the single tree concentrates importance in one or two features "
    "(whoever got picked for the root split), while the random forest distributes it "
    "more **evenly**. This is not because the random forest thinks all features are "
    "equally important -- it is because it has explored more possible tree structures "
    "and given every feature a fair audition."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 -- Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
st.header("5. Classification Results")

y_pred_rf = rf_model.predict(X_test)
metrics_rf = classification_metrics(y_test, y_pred_rf, labels=labels)

st.plotly_chart(
    plot_confusion_matrix(metrics_rf["confusion_matrix"], labels),
    use_container_width=True
)

st.markdown("**Per-class metrics:**")
report_df = pd.DataFrame(metrics_rf["report"]).T
report_df = report_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
report_df = report_df[["precision", "recall", "f1-score", "support"]].round(3)
st.dataframe(report_df, use_container_width=True)

warning_box(
    "Even with a random forest -- which is one of the strongest off-the-shelf classifiers "
    "in existence -- the Texas cities remain difficult to distinguish. There is no algorithmic "
    "magic that can overcome the fact that Dallas and Houston genuinely have similar weather. "
    "At some point, the data is the bottleneck, not the model."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 -- Individual Tree Variation
# ══════════════════════════════════════════════════════════════════════════════
st.header("6. Why Ensembles Work: Individual Tree Variation")

st.markdown(
    "To really understand why random forests work, you need to see how bad the individual "
    "trees are. Each one is trained on a random subset of data with a random subset of "
    "features, so each one is mediocre in its own special way. But collectively? "
    "They are a formidable committee."
)

if n_estimators >= 5:
    # Get predictions from individual trees
    n_show = min(20, n_estimators)
    individual_accs = []
    for i, tree in enumerate(rf_model.estimators_[:n_show]):
        acc = tree.score(X_test, y_test)
        individual_accs.append({"Tree": f"Tree {i+1}", "Accuracy": acc})

    indiv_df = pd.DataFrame(individual_accs)

    fig_indiv = go.Figure()
    fig_indiv.add_trace(go.Bar(
        x=indiv_df["Tree"], y=indiv_df["Accuracy"],
        marker_color="#7209B7", opacity=0.7, name="Individual Trees"
    ))
    fig_indiv.add_hline(
        y=rf_test_acc, line_dash="dash", line_color="#E63946",
        annotation_text=f"Ensemble: {rf_test_acc:.1%}"
    )
    fig_indiv.add_hline(
        y=indiv_df["Accuracy"].mean(), line_dash="dot", line_color="#2A9D8F",
        annotation_text=f"Mean: {indiv_df['Accuracy'].mean():.1%}"
    )
    apply_common_layout(fig_indiv, title=f"Individual Tree Accuracies (first {n_show} trees)", height=400)
    fig_indiv.update_layout(xaxis_title="Tree", yaxis_title="Test Accuracy")
    st.plotly_chart(fig_indiv, use_container_width=True)

    insight_box(
        f"Individual trees average **{indiv_df['Accuracy'].mean():.1%}** accuracy, "
        f"but the ensemble achieves **{rf_test_acc:.1%}**. The whole is greater than "
        "the sum of its parts -- literally. Each tree makes different mistakes in "
        "different directions, and when you average them, the mistakes cancel out "
        "while the signal reinforces. This is one of the most important ideas in all "
        "of machine learning."
    )
else:
    st.info("Increase n_estimators to at least 5 to see individual tree variation.")

code_example("""
from sklearn.ensemble import RandomForestClassifier

# Train random forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',     # random subset of features at each split
    min_samples_leaf=5,
    oob_score=True,          # free validation via out-of-bag samples
    n_jobs=-1,               # use all CPU cores
    random_state=42
)
rf.fit(X_train, y_train)

# OOB score (no validation set needed!)
print(f"OOB Score: {rf.oob_score_:.3f}")

# Feature importance (averaged across all trees)
importances = rf.feature_importances_

# Access individual trees
for tree in rf.estimators_[:5]:
    print(tree.score(X_test, y_test))
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 -- Quiz & Takeaways
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

quiz(
    "What are the two sources of randomness in a random forest?",
    [
        "Random learning rate and random depth",
        "Bootstrap sampling and random feature subsets at each split",
        "Random initialization and random class labels",
        "Random train/test split and random normalization",
    ],
    correct_idx=1,
    explanation="Bootstrap sampling gives each tree a different dataset; random feature subsets "
    "at each split ensure the trees are diverse. Both sources of randomness are necessary -- "
    "without feature randomization, all trees would make the same first split and end up "
    "highly correlated.",
    key="q_rf_1"
)

quiz(
    "What happens to a random forest when you add more trees?",
    [
        "It always overfits",
        "It always underfits",
        "Performance generally improves then plateaus -- more trees never hurts",
        "Performance decreases",
    ],
    correct_idx=2,
    explanation="This is one of the best properties of random forests: unlike single decision "
    "trees, you genuinely cannot overfit by adding more trees. Performance plateaus but "
    "never degrades. The only cost is computation time.",
    key="q_rf_2"
)

takeaways([
    "Random forests combine many decision trees through bootstrap aggregation (bagging) -- diverse mediocrity beats individual brilliance.",
    "Two sources of randomness: bootstrap samples and random feature subsets at each split. Both are needed for diverse, uncorrelated trees.",
    "More trees = more stable predictions. You cannot overfit a random forest by adding more trees -- just plateau.",
    "OOB (out-of-bag) error gives you a free estimate of generalization without needing a separate validation set.",
    "Feature importance averaged across all trees is more stable and trustworthy than a single tree's importance.",
    "Random forests significantly outperform single decision trees on our weather data, but still struggle with the Texas cities -- some problems require better data, not better models.",
])
