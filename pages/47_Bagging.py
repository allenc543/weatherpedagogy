"""Chapter 47: Bagging -- Bootstrap aggregation, variance reduction, OOB score."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.ml_helpers import prepare_classification_data, classification_metrics, plot_confusion_matrix
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(47, "Bagging (Bootstrap Aggregation)", part="XI")
st.markdown(
    "A single decision tree has a personality problem: it is brilliant but unreliable. "
    "Small changes in the training data can produce wildly different trees -- like "
    "asking the same question to different people at a party and getting contradictory "
    "answers from all of them. **Bagging** is the elegant fix: ask a hundred people, "
    "then take a vote. Train many trees on slightly different versions of the data and "
    "average their predictions. The individual trees are still noisy and opinionated, "
    "but their average is remarkably stable. Random Forest, the workhorse of applied ML, "
    "is bagging with one extra trick."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 47.1 How Bagging Works ──────────────────────────────────────────────────
st.header("47.1  How Bagging Works")

concept_box(
    "Bootstrap Aggregation",
    "<b>Bootstrap</b>: draw N samples <i>with replacement</i> from the training set. "
    "Because you are sampling with replacement, each bootstrap sample contains about "
    "63.2% unique rows -- the rest are duplicates. (Why 63.2%? Because the probability "
    "that any given row is NOT drawn in N tries is (1-1/N)^N, which approaches 1/e ~ 0.368 "
    "as N grows. So about 36.8% are left out, and 63.2% are included at least once. A fun "
    "bit of probability.)<br><br>"
    "<b>Aggregate</b>: train an independent model on each bootstrap sample. For "
    "classification, each model casts a vote and the majority wins. For regression, "
    "you just take the mean. Democracy in action."
)

formula_box(
    "Bagging Prediction (Classification)",
    r"\hat{y} = \text{mode}\bigl(\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_B\bigr)",
    "B = number of bootstrap models (estimators). Each one votes for a class, and the class with the most votes wins."
)

formula_box(
    "Out-of-Bag (OOB) Score",
    r"\text{OOB Error} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\!\bigl[\hat{y}_i^{oob} \neq y_i\bigr]",
    "Here is a clever trick: each sample is left out of about 36.8% of the bootstrap samples. So for each sample, you can get a prediction from the trees that did NOT train on it -- essentially a free cross-validation score, no held-out set required."
)

# Visual: bootstrap sampling
st.subheader("Bootstrap Sampling Illustration")
st.markdown(
    "Let us make this concrete. From an original dataset of indices [1, 2, 3, 4, 5], "
    "here are three possible bootstrap samples:"
)
rng = np.random.RandomState(42)
for b in range(3):
    sample = sorted(rng.choice(5, 5, replace=True) + 1)
    oob = sorted(set(range(1, 6)) - set(sample))
    st.markdown(f"**Bootstrap {b+1}:** {sample}  |  OOB (left out): {oob}")

# ── 47.2 Single Tree vs Bagged Trees ────────────────────────────────────────
st.header("47.2  Single Decision Tree vs Bagged Trees")

X_train, X_test, y_train, y_test, le, scaler = prepare_classification_data(
    fdf, FEATURE_COLS, target="city", test_size=0.2
)
city_labels = le.classes_

# Single tree
max_depth_tree = st.slider("Decision tree max_depth", 1, 30, 10, key="single_depth")
single_tree = DecisionTreeClassifier(max_depth=max_depth_tree, random_state=42)
single_tree.fit(X_train, y_train)
single_acc = accuracy_score(y_test, single_tree.predict(X_test))
single_train_acc = accuracy_score(y_train, single_tree.predict(X_train))

# Bagged trees
n_estimators = st.slider("Number of bagged trees (n_estimators)", 2, 200, 50, 5, key="bag_n_est")

bagged = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=max_depth_tree),
    n_estimators=n_estimators, random_state=42, oob_score=True, n_jobs=-1,
)
bagged.fit(X_train, y_train)
bagged_acc = accuracy_score(y_test, bagged.predict(X_test))
bagged_train_acc = accuracy_score(y_train, bagged.predict(X_train))
oob_score = bagged.oob_score_

col1, col2, col3 = st.columns(3)
col1.metric("Single Tree Test Acc", f"{single_acc:.4f}")
col2.metric("Bagged Trees Test Acc", f"{bagged_acc:.4f}", delta=f"{bagged_acc - single_acc:+.4f}")
col3.metric("OOB Score", f"{oob_score:.4f}")

comp_df = pd.DataFrame({
    "Model": ["Single Decision Tree", f"Bagged ({n_estimators} trees)"],
    "Train Accuracy": [single_train_acc, bagged_train_acc],
    "Test Accuracy": [single_acc, bagged_acc],
    "Gap (Overfit)": [single_train_acc - single_acc, bagged_train_acc - bagged_acc],
})
st.dataframe(comp_df.style.format({
    "Train Accuracy": "{:.4f}", "Test Accuracy": "{:.4f}", "Gap (Overfit)": "{:.4f}"
}), use_container_width=True, hide_index=True)

insight_box(
    "Look at the overfitting gap (train minus test accuracy). The single tree has a "
    "big gap -- it has memorized training-set-specific noise. The bagged ensemble has "
    "a smaller gap because averaging many noisy models cancels out their individual "
    "hallucinations. This is variance reduction in action: the wisdom of the crowd, "
    "applied to decision trees."
)

# ── 47.3 Variance Reduction with More Estimators ────────────────────────────
st.header("47.3  How Variance Decreases with More Estimators")

concept_box(
    "Variance Reduction",
    "As you add more trees to the ensemble, the variance of the combined prediction "
    "decreases, thanks to the law of large numbers. However -- and this is important -- "
    "bagging does <b>not</b> reduce bias. If each individual tree is biased (say, "
    "max_depth=1), averaging a thousand of them still gives a biased answer. Bagging "
    "is the 'wisdom of crowds' approach: it works great when each member of the crowd "
    "is individually noisy but on-average correct."
)

st.subheader("Test Accuracy vs Number of Estimators")
est_values = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200]
est_accs = []
est_oobs = []

# Subsample for speed
sample_n = min(5000, len(X_train))
rng = np.random.RandomState(42)
idx = rng.choice(len(X_train), sample_n, replace=False)
X_tr_s = X_train.iloc[idx]
y_tr_s = y_train[idx]

with st.spinner("Training ensembles of different sizes..."):
    for n in est_values:
        bag = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=max_depth_tree),
            n_estimators=n, random_state=42, oob_score=True, n_jobs=-1,
        )
        bag.fit(X_tr_s, y_tr_s)
        est_accs.append(accuracy_score(y_test, bag.predict(X_test)))
        est_oobs.append(bag.oob_score_)

fig_est = go.Figure()
fig_est.add_trace(go.Scatter(
    x=est_values, y=est_accs, mode="lines+markers",
    name="Test Accuracy", line=dict(color="#E63946"),
))
fig_est.add_trace(go.Scatter(
    x=est_values, y=est_oobs, mode="lines+markers",
    name="OOB Score", line=dict(color="#2A9D8F", dash="dash"),
))
fig_est.add_hline(y=single_acc, line_dash="dot", line_color="gray",
                  annotation_text=f"Single Tree: {single_acc:.3f}")
apply_common_layout(fig_est, title="Accuracy vs Number of Estimators", height=450)
fig_est.update_layout(xaxis_title="Number of Estimators", yaxis_title="Accuracy")
st.plotly_chart(fig_est, use_container_width=True)

insight_box(
    "Accuracy improves rapidly with the first few trees (going from 2 to 20 is a big "
    "deal), then plateaus. Beyond about 50-100 trees, you are in diminishing returns "
    "territory. But here is the beautiful thing about bagging: more trees never hurts. "
    "You do not overfit by adding more estimators. The only cost is computation time. "
    "This is quite unlike boosting, as we will see in the next chapter."
)

# ── 47.4 Individual Tree Variance ───────────────────────────────────────────
st.header("47.4  Variability of Individual Trees")

st.markdown(
    "Each tree in the ensemble makes different errors because it trained on a different "
    "bootstrap sample. Let us look at how much their individual accuracies vary -- "
    "and then marvel at the fact that their average is better than any one of them."
)

n_trees_to_show = min(20, n_estimators)
individual_accs = []
for est in bagged.estimators_[:n_trees_to_show]:
    pred = est.predict(scaler.transform(X_test) if scaler else X_test)
    individual_accs.append(accuracy_score(y_test, pred))

fig_var = go.Figure()
fig_var.add_trace(go.Bar(
    x=[f"Tree {i+1}" for i in range(n_trees_to_show)],
    y=individual_accs,
    marker_color="#264653",
))
fig_var.add_hline(y=bagged_acc, line_dash="dash", line_color="#E63946",
                  annotation_text=f"Ensemble: {bagged_acc:.3f}")
fig_var.add_hline(y=np.mean(individual_accs), line_dash="dot", line_color="#2A9D8F",
                  annotation_text=f"Mean Individual: {np.mean(individual_accs):.3f}")
apply_common_layout(fig_var, title=f"Individual Tree Accuracies (first {n_trees_to_show})", height=400)
fig_var.update_layout(yaxis_title="Accuracy", xaxis_title="Tree")
st.plotly_chart(fig_var, use_container_width=True)

var_stats = pd.DataFrame({
    "Metric": ["Mean individual accuracy", "Std of individual accuracies", "Ensemble accuracy", "Variance reduction"],
    "Value": [
        f"{np.mean(individual_accs):.4f}",
        f"{np.std(individual_accs):.4f}",
        f"{bagged_acc:.4f}",
        f"{np.mean(individual_accs) - bagged_acc:+.4f} (ensemble gains)",
    ],
})
st.dataframe(var_stats, use_container_width=True, hide_index=True)

insight_box(
    "Notice that the ensemble accuracy is typically higher than even the *average* "
    "individual tree accuracy. This is not magic -- it is what happens when you "
    "aggregate diverse estimators. Where one tree goes left and another goes right, "
    "the ensemble hedges its bets and often ends up closer to the truth. Condorcet "
    "proved this for juries in 1785. We are just applying it to decision trees."
)

# ── 47.5 Random Forest as Advanced Bagging ──────────────────────────────────
st.header("47.5  Random Forest: Bagging with Feature Randomness")

concept_box(
    "Random Forest = Bagging + Random Feature Subsets",
    "Wait, you might say, if bagging is so great, why does anyone bother with Random "
    "Forest? Because there is a subtle problem with plain bagging: if one feature is "
    "very strong, every tree will split on that feature first, making the trees "
    "correlated. Correlated trees help less when averaged. Random Forest fixes this "
    "by only considering a <b>random subset</b> of features at each split. This "
    "decorrelates the trees, making the ensemble even more effective. It is the "
    "difference between asking 100 people the same question and asking 100 people "
    "100 different questions and combining their answers."
)

rf = RandomForestClassifier(
    n_estimators=n_estimators, max_depth=max_depth_tree, random_state=42, n_jobs=-1,
    oob_score=True,
)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
rf_oob = rf.oob_score_

final_comp = pd.DataFrame({
    "Model": ["Single Decision Tree", f"Bagging ({n_estimators} trees)", f"Random Forest ({n_estimators} trees)"],
    "Test Accuracy": [single_acc, bagged_acc, rf_acc],
    "OOB Score": ["-", f"{oob_score:.4f}", f"{rf_oob:.4f}"],
})
st.dataframe(final_comp, use_container_width=True, hide_index=True)

fig_final = px.bar(final_comp, x="Model", y="Test Accuracy",
                   color="Model", color_discrete_sequence=["#264653", "#2A9D8F", "#E63946"],
                   title="Single Tree vs Bagging vs Random Forest")
apply_common_layout(fig_final, height=400)
st.plotly_chart(fig_final, use_container_width=True)

code_example("""from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=10),
    n_estimators=100, oob_score=True, random_state=42
)
bag.fit(X_train, y_train)
print(f"OOB Score: {bag.oob_score_:.4f}")

# Random Forest (bagging + feature randomness)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, oob_score=True)
rf.fit(X_train, y_train)
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "What percentage of unique samples does each bootstrap sample contain on average?",
    [
        "50%",
        "63.2%",
        "75%",
        "100%",
    ],
    correct_idx=1,
    explanation="Each sample has a (1 - 1/n)^n probability of NOT being selected, which approaches 1/e as n grows large. So about 36.8% are missed and 63.2% are included. This is one of those results where the number e shows up in a place you would not expect, which is always a good sign.",
    key="ch47_quiz1",
)

quiz(
    "Bagging primarily reduces:",
    [
        "Bias",
        "Variance",
        "Irreducible error",
        "Training time",
    ],
    correct_idx=1,
    explanation="Bagging averages many high-variance models, reducing the variance of the ensemble. It does not help with bias -- if each tree is too simple to capture the pattern, averaging a thousand of them will not fix that.",
    key="ch47_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "**Bagging** trains multiple models on bootstrapped samples and aggregates their predictions. It is the 'ask 100 people and take a vote' approach to machine learning.",
    "It primarily reduces **variance** -- making predictions more stable by averaging out the noise in individual models.",
    "The **OOB score** gives you a free validation estimate without needing a separate test set. About 36.8% of data is left out of each bootstrap sample, and you use those leftovers for evaluation.",
    "More estimators always help (or at least never hurt), with diminishing returns beyond about 50-100 trees. The only cost is computation time.",
    "**Random Forest** extends bagging by randomizing feature subsets at each split, which decorrelates the trees and makes the ensemble even stronger.",
    "The ensemble is stronger than any individual tree because aggregation cancels out random errors. This is Condorcet's jury theorem, applied to decision trees.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 46: Regression Metrics",
    prev_page="46_Regression_Metrics.py",
    next_label="Ch 48: Boosting",
    next_page="48_Boosting.py",
)
