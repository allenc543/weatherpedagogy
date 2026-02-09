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
    "Let me set up the problem first. We are trying to predict which of 6 cities a "
    "weather reading came from, using temperature, humidity, wind speed, and pressure. "
    "In the last few chapters, we used a single decision tree for this. The tree looks "
    "at the data and builds a set of rules: 'If temperature > 30 C and humidity > 70%, "
    "predict Houston. If temperature < 5 C, predict NYC.' Reasonable enough."
)
st.markdown(
    "But here is the problem with a single decision tree: it is *unstable*. If you "
    "trained it on a slightly different subset of the data -- say, you happened to "
    "include a few more Dallas readings and a few fewer Austin readings -- you would "
    "get a completely different tree with different splits and different predictions. "
    "It might decide that temperature > 31 C (not 30) is the Houston cutoff, or it "
    "might split on pressure first instead of temperature. Small changes in the data "
    "produce wildly different models. This is the high-variance problem from Chapter 44."
)
st.markdown(
    "**Bagging** is the fix: instead of asking one unstable tree, train 100 trees on "
    "slightly different versions of the data and let them vote. Each individual tree is "
    "still noisy and opinionated, but their *collective* answer is remarkably stable."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 47.1 How Bagging Works ──────────────────────────────────────────────────
st.header("47.1  How Bagging Works")

st.markdown(
    "The idea has two parts -- the 'bootstrap' and the 'aggregate' -- and both are "
    "simple once you see them with our weather data."
)

concept_box(
    "Bootstrap: Creating Different Versions of the Training Data",
    "Suppose our training set has 80,000 weather readings. To create one 'bootstrap "
    "sample,' you randomly draw 80,000 readings <i>with replacement</i> from the "
    "original 80,000. With replacement means the same reading can be picked more than "
    "once. Some readings appear twice, some three times, and some are never picked at "
    "all.<br><br>"
    "How many readings get left out? It turns out to be about 36.8%, because the "
    "probability that any single reading is never chosen in 80,000 draws is "
    "(1 - 1/80000)^80000, which approaches 1/e = 0.368 as the dataset grows. "
    "So each bootstrap sample uses about 63.2% of the original data. The 36.8% "
    "left out are called the <b>out-of-bag (OOB)</b> samples -- and they give us "
    "a free validation set, as we will see shortly.<br><br>"
    "<b>Aggregate</b>: Train an independent decision tree on each bootstrap sample. "
    "Now you have 100 trees, each trained on a slightly different slice of the data. "
    "For a new weather reading, each tree votes for a city. The city with the most "
    "votes wins. That is it. Democracy applied to decision trees."
)

formula_box(
    "Bagging Prediction (Classification)",
    r"\hat{y} = \text{mode}\bigl(\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_B\bigr)",
    "B = number of bootstrap trees. Each tree votes for a city (Dallas, Houston, NYC...) and the city with the most votes wins. If 67 trees say 'Houston' and 33 say 'Dallas,' the ensemble predicts Houston."
)

formula_box(
    "Out-of-Bag (OOB) Score",
    r"\text{OOB Error} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\!\bigl[\hat{y}_i^{oob} \neq y_i\bigr]",
    "Each weather reading is left out of roughly 36.8% of the trees. For that reading, "
    "you can get a prediction using only the trees that did NOT train on it -- a free "
    "cross-validation score without setting aside a separate test set. The OOB score "
    "tells you how well the ensemble generalizes to data it has not seen."
)

# Visual: bootstrap sampling
st.subheader("Bootstrap Sampling Illustration")
st.markdown(
    "Let me make this concrete with a tiny example. Imagine we had only 5 weather "
    "readings, labeled 1 through 5. Here are three bootstrap samples -- notice how "
    "some readings repeat and some are left out entirely:"
)
rng = np.random.RandomState(42)
for b in range(3):
    sample = sorted(rng.choice(5, 5, replace=True) + 1)
    oob = sorted(set(range(1, 6)) - set(sample))
    st.markdown(f"**Bootstrap {b+1}:** {sample}  |  OOB (left out): {oob}")

st.markdown(
    "In Bootstrap 1, reading 4 appears twice and readings like 2 or 3 might be "
    "missing. The tree trained on Bootstrap 1 has never seen those OOB readings, so "
    "we can use them as a mini-test set for that specific tree. Across all trees, "
    "every reading ends up in the OOB set for about a third of the trees."
)

# ── 47.2 Single Tree vs Bagged Trees ────────────────────────────────────────
st.header("47.2  Single Decision Tree vs Bagged Trees")

st.markdown(
    "Now let us see bagging in action on our city classification task. We will train "
    "a single decision tree and a bagged ensemble side by side, and compare accuracy "
    "and overfitting."
)

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

st.markdown(
    f"**max_depth slider**: This controls how many questions each tree can ask. "
    f"max_depth=1 means one question ('Is temperature > 20 C?'). max_depth=10 means "
    f"10 nested questions. Higher depth = more complex tree = more potential to memorize "
    f"noise. Try dragging it and watching the overfitting gap change."
)

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
    f"Look at the 'Gap (Overfit)' column. The single tree has a training accuracy of "
    f"{single_train_acc:.4f} and a test accuracy of {single_acc:.4f} -- a gap of "
    f"{single_train_acc - single_acc:.4f}. It memorized specific weather readings in the "
    f"training set (like 'when temperature is exactly 22.7 C and humidity is 68.3%, that "
    f"is Dallas reading #4,721'). The bagged ensemble has a smaller gap "
    f"({bagged_train_acc - bagged_acc:.4f}) because averaging 100 noisy opinions cancels "
    f"out their individual hallucinations. This is variance reduction in action: no single "
    f"tree is wise, but the crowd is."
)

# ── 47.3 Variance Reduction with More Estimators ────────────────────────────
st.header("47.3  How Variance Decreases with More Estimators")

st.markdown(
    "How many trees do you actually need? One tree is unstable. Two trees are better. "
    "Ten trees are much better. But does going from 100 to 200 trees help? Let us find out."
)

concept_box(
    "Variance Reduction: Why Averaging Works",
    "Imagine 100 weather forecasters each independently predicting Dallas's temperature "
    "for tomorrow. Each forecaster is individually noisy -- one says 32 C, another says "
    "28 C, another says 35 C. But their average will be close to the true temperature, "
    "because the random errors tend to cancel out. That is the law of large numbers, "
    "and it is exactly what bagging exploits.<br><br>"
    "But -- and this is important -- bagging does <b>not</b> reduce bias. If each "
    "individual tree has max_depth=1 (it can only ask one question), it is too simple "
    "to distinguish 6 cities. Averaging 1,000 trees that each ask only one question "
    "still gives you a model that can only ask one question. Bagging helps when each "
    "tree is individually powerful but noisy, not when each tree is structurally too "
    "simple."
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
    "Notice the shape of the curve. Going from 1 tree to 10 trees is a big improvement -- "
    "you go from one unstable opinion to a small committee. Going from 10 to 50 still "
    "helps noticeably. But going from 100 to 200? The curve has mostly flattened. You "
    "are in diminishing-returns territory. And here is the beautiful thing about bagging: "
    "more trees *never* hurts. Unlike boosting (next chapter), you cannot overfit by "
    "adding more estimators. The only cost is computation time. So the practical advice "
    "is: start with 100 trees, and if you have the compute budget, go to 500. You will "
    "never regret it."
)

# ── 47.4 Individual Tree Variance ───────────────────────────────────────────
st.header("47.4  Variability of Individual Trees")

st.markdown(
    "Let me show you something that makes the 'wisdom of crowds' intuition really "
    "click. Each tree in the bagging ensemble was trained on a different bootstrap "
    "sample of weather readings. Let us look at how their individual accuracies "
    "scatter -- and then marvel at the fact that their collective answer is better "
    "than any one of them."
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
    "Look at the bar chart. Some trees are better at identifying certain cities, "
    "others are worse. Tree 3 might be great at spotting NYC weather (it trained on "
    "a bootstrap sample with lots of cold-weather readings) but terrible at "
    "distinguishing Dallas from San Antonio. Tree 7 might be the opposite. When they "
    "vote together, their individual blind spots get outvoted. Tree 3's bad Dallas "
    "guesses get overruled by Trees 1, 5, and 12, which happened to train on data "
    "that gave them better Dallas intuition. The ensemble is typically better than "
    "even the *best* individual tree, because it combines their complementary strengths."
)

# ── 47.5 Random Forest as Advanced Bagging ──────────────────────────────────
st.header("47.5  Random Forest: Bagging with Feature Randomness")

st.markdown(
    "If bagging is so effective, why does Random Forest exist? What does it add?"
)

concept_box(
    "The Correlation Problem That Random Forest Solves",
    "In our weather dataset, temperature is by far the most powerful feature for "
    "distinguishing cities. NYC has cold winters; Texas cities are hot. So when you "
    "bag 100 decision trees, every single tree splits on temperature first. They all "
    "make the same first decision, which means they are <b>correlated</b> -- they tend "
    "to make the same mistakes on the same readings.<br><br>"
    "Averaging correlated predictions does not help as much as averaging uncorrelated "
    "ones. (Think of it this way: polling 100 people who all read the same newspaper "
    "gives you less information than polling 100 people who read different sources.)<br><br>"
    "Random Forest fixes this by only considering a <b>random subset</b> of features at "
    "each split. Maybe Tree 1 can only choose between humidity and wind speed at the "
    "first split. Tree 2 gets temperature and pressure. Tree 3 gets wind speed and "
    "pressure. Now the trees are forced to find different patterns in the data. Some "
    "discover that LA has uniquely low humidity. Others learn that NYC has distinctive "
    "pressure patterns. The trees are <b>decorrelated</b>, and their combined vote is "
    "even more powerful than plain bagging."
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

st.markdown(
    f"Random Forest ({rf_acc:.4f}) typically edges out plain Bagging ({bagged_acc:.4f}) "
    f"by a small but consistent margin. On our weather data, the improvement might be "
    f"modest because we only have 4 input features (temperature, humidity, wind, pressure), "
    f"so there is less room for feature randomization. On datasets with 50 or 100 features, "
    f"the Random Forest advantage becomes much more pronounced."
)

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
    "What percentage of unique weather readings does each bootstrap sample contain on average?",
    [
        "50%",
        "63.2%",
        "75%",
        "100%",
    ],
    correct_idx=1,
    explanation=(
        "When you draw 80,000 readings with replacement from 80,000, each reading has a "
        "(1 - 1/80000)^80000 probability of never being chosen, which approaches 1/e = "
        "36.8%. So about 36.8% are left out and 63.2% appear at least once. The 36.8% "
        "that are left out form the OOB (out-of-bag) set for that particular tree -- "
        "readings the tree has never trained on, which can be used as a free validation set."
    ),
    key="ch47_quiz1",
)

quiz(
    "Bagging primarily reduces:",
    [
        "Bias -- making too-simple models more powerful",
        "Variance -- making unstable models more stable",
        "Irreducible error -- reducing noise in the data",
        "Training time -- making models faster to train",
    ],
    correct_idx=1,
    explanation=(
        "Bagging averages many high-variance models to reduce the variance of the "
        "ensemble. Each individual tree on our weather data might give a different "
        "answer for an ambiguous Dallas-vs-San-Antonio reading, but their average "
        "converges on the more likely answer. It does NOT help with bias -- if each "
        "tree is max_depth=1 (too simple to distinguish 6 cities), averaging 1,000 "
        "of them still gives a too-simple answer."
    ),
    key="ch47_quiz2",
)

quiz(
    "Why does Random Forest add random feature subsets on top of bagging?",
    [
        "To make training faster by using fewer features",
        "To decorrelate the trees so their combined vote is more powerful",
        "To reduce the number of trees needed",
        "To perform automatic feature selection",
    ],
    correct_idx=1,
    explanation=(
        "In plain bagging on our weather data, every tree splits on temperature first "
        "because it is the strongest signal. This makes the trees correlated -- they all "
        "make similar mistakes. Random Forest forces each tree to consider only a random "
        "subset of features at each split, so some trees learn to identify cities using "
        "humidity patterns, others using pressure, others using wind. The decorrelated "
        "trees bring genuinely different perspectives, making the ensemble more effective "
        "than a crowd of people who all read the same newspaper."
    ),
    key="ch47_quiz3",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "**Bagging** trains many decision trees on bootstrapped samples of weather data and lets them vote. It is the 'ask 100 forecasters and take the consensus' approach.",
    "It primarily reduces **variance** -- the tendency of a single tree to give wildly different answers depending on which weather readings it trained on. The ensemble's average is much more stable.",
    "The **OOB score** is a free validation estimate: about 36.8% of readings are left out of each tree's training set, so you can test each tree on data it has never seen.",
    "More trees always help (or at least never hurt), with diminishing returns beyond about 50-100. Unlike boosting, you cannot overfit by adding more trees.",
    "**Random Forest** extends bagging by forcing each tree to consider random feature subsets. This decorrelates the trees -- instead of all splitting on temperature first, some discover patterns in humidity, wind, or pressure.",
    "The ensemble is stronger than any individual tree because voting cancels out random errors. Where Tree 3 confuses Dallas for San Antonio, Trees 5 and 12 get it right and outvote the mistake.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 46: Regression Metrics",
    prev_page="46_Regression_Metrics.py",
    next_label="Ch 48: Boosting",
    next_page="48_Boosting.py",
)
