"""Chapter 22: Decision Trees -- Splits, Gini, entropy, and interpretable classification."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, export_text

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.ml_helpers import prepare_classification_data, classification_metrics, plot_confusion_matrix
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(22, "Decision Trees", part="V")
st.markdown(
    "Here is what is delightful about decision trees: they classify data the same way "
    "you would if someone handed you a stack of weather reports and asked you to sort "
    "them by city. 'Is the temperature below 10 degrees? Then it is probably NYC. Is "
    "the humidity above 80%? Then maybe Houston.' Decision trees learn exactly these "
    "kinds of **if/then rules** from data, they are easy to interpret, and -- as we "
    "will see in the next chapter -- they are the building blocks for some of the most "
    "powerful algorithms in machine learning."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
filt = sidebar_filters(df)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 -- How Splits Work
# ══════════════════════════════════════════════════════════════════════════════
st.header("1. How Decision Trees Make Splits")

concept_box(
    "Recursive Binary Splitting",
    "At each node, the tree asks a very specific question: <i>Which feature, and which "
    "threshold on that feature, best separates the classes?</i> It tries every feature "
    "and every possible threshold, picks the split that maximally reduces impurity "
    "(more on that in a moment), and then recursively does the same thing on each child "
    "node. It is a greedy algorithm -- it makes the best local choice at each step and "
    "never looks back."
)

st.markdown(
    "For our weather data, the first split is almost always on **temperature** or "
    "**surface pressure**. This should not surprise you: if you were sorting weather "
    "reports by city with your eyes closed, temperature is the first thing you would "
    "check too."
)

# Show distribution that motivates the first split
st.subheader("Why Temperature is Often the First Split")
sample_plot = filt.sample(min(5000, len(filt)), random_state=42)
fig_hist = px.histogram(
    sample_plot, x="temperature_c", color="city",
    color_discrete_map=CITY_COLORS, barmode="overlay", nbins=60,
    opacity=0.6, title="Temperature Distribution by City",
    labels=FEATURE_LABELS,
)
apply_common_layout(fig_hist, title="Temperature Distribution by City", height=400)
st.plotly_chart(fig_hist, use_container_width=True)

insight_box(
    "Look at this histogram. LA's temperature distribution barely overlaps with NYC's. "
    "A single threshold -- say, 'is the temperature below 5 degrees?' -- would already "
    "separate a huge chunk of NYC data from everything else. The tree discovers these "
    "thresholds automatically, which is really just the algorithm rediscovering things "
    "that would be obvious to a meteorologist."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 -- Gini Impurity vs Entropy
# ══════════════════════════════════════════════════════════════════════════════
st.header("2. Splitting Criteria: Gini Impurity vs Entropy")

col_g, col_e = st.columns(2)
with col_g:
    formula_box(
        "Gini Impurity",
        r"G = 1 - \sum_{k=1}^{K} p_k^2",
        "Imagine picking two random samples from a node. Gini impurity is the probability "
        "that they would have different labels. Ranges from 0 (everyone agrees, pure node) "
        "to 1 - 1/K (maximum chaos)."
    )
with col_e:
    formula_box(
        "Entropy (Information Gain)",
        r"H = -\sum_{k=1}^{K} p_k \log_2(p_k)",
        "Borrowed from information theory. How many bits do you need to encode which class "
        "a random sample belongs to? A pure node needs 0 bits. Maximum entropy is when you "
        "genuinely have no idea."
    )

# Interactive impurity comparison
st.subheader("Impurity Comparison (Binary Case)")
p_slider = st.slider("Proportion of class 1 (p)", 0.0, 1.0, 0.5, 0.01, key="impurity_p")

p_vals = np.linspace(0.001, 0.999, 200)
gini_vals = 1 - p_vals**2 - (1 - p_vals)**2
entropy_vals = -(p_vals * np.log2(p_vals) + (1 - p_vals) * np.log2(1 - p_vals))

fig_imp = go.Figure()
fig_imp.add_trace(go.Scatter(x=p_vals, y=gini_vals, name="Gini", line=dict(color="#E63946", width=3)))
fig_imp.add_trace(go.Scatter(x=p_vals, y=entropy_vals, name="Entropy", line=dict(color="#2E86C1", width=3)))
# Mark current slider position
gini_at_p = 1 - p_slider**2 - (1 - p_slider)**2
entropy_at_p = -(p_slider * np.log2(p_slider) + (1 - p_slider) * np.log2(1 - p_slider)) if 0 < p_slider < 1 else 0
fig_imp.add_trace(go.Scatter(x=[p_slider], y=[gini_at_p], mode="markers", name=f"Gini={gini_at_p:.3f}",
                              marker=dict(color="#E63946", size=12)))
fig_imp.add_trace(go.Scatter(x=[p_slider], y=[entropy_at_p], mode="markers", name=f"Entropy={entropy_at_p:.3f}",
                              marker=dict(color="#2E86C1", size=12)))
apply_common_layout(fig_imp, title="Gini vs Entropy as a Function of Class Proportion", height=400)
fig_imp.update_layout(xaxis_title="p (proportion of class 1)", yaxis_title="Impurity")
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown(
    f"At p = {p_slider:.2f}: **Gini = {gini_at_p:.4f}**, **Entropy = {entropy_at_p:.4f}**. "
    "Both metrics peak at p = 0.5 (maximum uncertainty -- you are basically flipping "
    "a coin) and bottom out at p = 0 or 1 (pure certainty). In practice, Gini and "
    "entropy almost always pick the same splits, so the choice between them is about "
    "as consequential as choosing between Pepsi and Coke. Computer scientists will "
    "argue about this forever, but the data mostly does not care."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 -- Interactive Decision Tree
# ══════════════════════════════════════════════════════════════════════════════
st.header("3. Train a Decision Tree")

st.markdown(
    "Here is where it gets fun. Use the depth slider below and watch what happens. "
    "A depth-1 tree can ask exactly one question. A depth-20 tree can ask twenty "
    "questions in sequence, which turns out to be enough to memorize the training "
    "data almost perfectly -- and that is exactly the problem."
)

max_depth = st.slider("Maximum tree depth", 1, 20, 5, 1, key="dt_depth")
criterion = st.selectbox("Splitting criterion", ["gini", "entropy"], key="dt_criterion")
min_samples_leaf = st.slider("Minimum samples per leaf", 1, 100, 5, 1, key="dt_leaf")

X_train, X_test, y_train, y_test, le, scaler = prepare_classification_data(
    filt, FEATURE_COLS, target="city", test_size=0.2, scale=False, seed=42
)

dt_model = DecisionTreeClassifier(
    max_depth=max_depth,
    criterion=criterion,
    min_samples_leaf=min_samples_leaf,
    random_state=42
)
dt_model.fit(X_train, y_train)
y_pred_train = dt_model.predict(X_train)
y_pred_test = dt_model.predict(X_test)
labels = le.classes_.tolist()

metrics_train = classification_metrics(y_train, y_pred_train, labels=labels)
metrics_test = classification_metrics(y_test, y_pred_test, labels=labels)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Train Accuracy", f"{metrics_train['accuracy']:.1%}")
col2.metric("Test Accuracy", f"{metrics_test['accuracy']:.1%}")
gap = metrics_train["accuracy"] - metrics_test["accuracy"]
col3.metric("Overfitting Gap", f"{gap:.1%}")
col4.metric("Tree Depth", max_depth)

if gap > 0.05:
    warning_box(
        f"The train-test gap is {gap:.1%}. This is the classic decision tree failure mode: "
        "the tree has memorized quirks of the training data rather than learning general "
        "patterns. It is like a student who memorized the answer key instead of understanding "
        "the material -- great scores on practice tests, terrible on the real exam. "
        "Try reducing max_depth or increasing min_samples_leaf."
    )

st.plotly_chart(
    plot_confusion_matrix(metrics_test["confusion_matrix"], labels),
    use_container_width=True
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 -- Pruning & Depth Analysis
# ══════════════════════════════════════════════════════════════════════════════
st.header("4. Pruning: Finding the Right Depth")

concept_box(
    "Pre-pruning vs Post-pruning",
    "An unpruned decision tree will keep splitting until every single leaf is pure, "
    "which means it has essentially memorized the training set. This is the ML equivalent "
    "of a conspiracy theory: it finds a pattern for everything, even the noise.<br><br>"
    "<b>Pre-pruning</b> stops the tree early (max_depth, min_samples). <b>Post-pruning</b> "
    "grows the full tree, then trims branches that do not help on validation data. Both are "
    "ways of saying 'please do not memorize the noise.'"
)

st.subheader("Accuracy vs Tree Depth")

depth_range = range(1, 21)
train_accs = []
test_accs = []
for d in depth_range:
    dt_temp = DecisionTreeClassifier(max_depth=d, min_samples_leaf=5, random_state=42)
    dt_temp.fit(X_train, y_train)
    train_accs.append(dt_temp.score(X_train, y_train))
    test_accs.append(dt_temp.score(X_test, y_test))

depth_df = pd.DataFrame({
    "Depth": list(depth_range),
    "Train Accuracy": train_accs,
    "Test Accuracy": test_accs,
})
depth_melt = depth_df.melt(id_vars="Depth", var_name="Set", value_name="Accuracy")

fig_depth = px.line(
    depth_melt, x="Depth", y="Accuracy", color="Set",
    title="Train vs Test Accuracy by Tree Depth",
    color_discrete_map={"Train Accuracy": "#E63946", "Test Accuracy": "#2E86C1"},
    markers=True
)
apply_common_layout(fig_depth, title="Train vs Test Accuracy by Tree Depth", height=400)
st.plotly_chart(fig_depth, use_container_width=True)

best_depth = depth_df.loc[depth_df["Test Accuracy"].idxmax(), "Depth"]
best_acc = depth_df["Test Accuracy"].max()
insight_box(
    f"The best test accuracy is **{best_acc:.1%}** at depth **{int(best_depth)}**. "
    "After that, the red line (train accuracy) keeps climbing toward 100% while "
    "the blue line (test accuracy) stalls or drops. This is the textbook picture of "
    "overfitting, and it is worth staring at until you can recognize it in your sleep. "
    "Every time someone complains that their model 'works great in testing but fails "
    "in production,' this is what happened."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 -- Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
st.header("5. Feature Importance")

concept_box(
    "How Feature Importance is Computed",
    "Every time the tree makes a split on a feature, it reduces impurity by some amount. "
    "Feature importance is just the total impurity reduction from all splits on that feature, "
    "normalized so they sum to 1. Features used higher in the tree (closer to the root) tend "
    "to be more important, because they separate the most data. The root split is the tree's "
    "best idea about what matters most."
)

importances = dt_model.feature_importances_
feat_imp_df = pd.DataFrame({
    "Feature": [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
    "Importance": importances
}).sort_values("Importance", ascending=True)

fig_fi = px.bar(
    feat_imp_df, x="Importance", y="Feature", orientation="h",
    title="Decision Tree Feature Importance",
    color="Importance", color_continuous_scale="Viridis"
)
apply_common_layout(fig_fi, title="Decision Tree Feature Importance", height=350)
st.plotly_chart(fig_fi, use_container_width=True)

top_feature = feat_imp_df.iloc[-1]["Feature"]
insight_box(
    f"The most important feature is **{top_feature}**. This makes sense: the root of "
    "the tree -- the very first question it asks -- uses whatever feature best divides "
    "the data, and everything downstream is carved from whatever is left. A decision "
    "tree is basically an algorithm that asks 'what is the single most useful question "
    "I could ask right now?' and repeats."
)

# Show tree rules (top levels)
st.subheader("Tree Rules (First 3 Levels)")
tree_text = export_text(dt_model, feature_names=FEATURE_COLS, max_depth=3)
st.code(tree_text, language="text")

code_example("""
from sklearn.tree import DecisionTreeClassifier, export_text

# Train decision tree
dt = DecisionTreeClassifier(max_depth=5, criterion='gini', min_samples_leaf=5)
dt.fit(X_train, y_train)

# Feature importance
importances = dt.feature_importances_

# View tree rules
print(export_text(dt, feature_names=feature_names, max_depth=3))
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 -- Quiz & Takeaways
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

quiz(
    "What happens when you increase a decision tree's max_depth too much?",
    [
        "The model underfits",
        "The model overfits -- training accuracy goes up but test accuracy may drop",
        "The model becomes faster to train",
        "The model ignores some features",
    ],
    correct_idx=1,
    explanation="A deep tree memorizes the training data, noise and all. It is the ML "
    "equivalent of studying only the answer key: perfect on the practice exam, terrible "
    "on anything new. Pruning is how you tell it to learn the material instead.",
    key="q_dt_1"
)

quiz(
    "A Gini impurity of 0 means:",
    [
        "Maximum impurity -- all classes are equally likely",
        "The node is pure -- all samples belong to one class",
        "The tree has no splits",
        "The model has 100% test accuracy",
    ],
    correct_idx=1,
    explanation="Gini = 0 means every sample in that node belongs to the same class. "
    "The tree has no uncertainty about what to predict for points landing here.",
    key="q_dt_2"
)

takeaways([
    "Decision trees learn a sequence of if/then rules by greedily splitting on the feature and threshold that most reduces impurity.",
    "Gini impurity and entropy are both valid splitting criteria -- they usually produce nearly identical trees, so do not lose sleep over the choice.",
    "Deeper trees have more capacity but risk overfitting. The train-test gap is your canary in the coal mine.",
    "Feature importance tells you which features the tree relied on most. The root split is the most informative single question.",
    "Temperature and pressure are often the first splits because they most sharply separate cities.",
    "Decision trees require no feature scaling -- a rare and pleasant property that logistic regression and SVM cannot claim.",
])
