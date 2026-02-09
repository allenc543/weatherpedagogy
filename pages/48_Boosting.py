"""Chapter 48: Boosting (XGBoost/LightGBM) -- Sequential learning, learning rate, early stopping."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
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
chapter_header(48, "Boosting (XGBoost / LightGBM)", part="XI")
st.markdown(
    "Bagging asks a hundred independent models for their opinion and takes a vote. "
    "Boosting does something more interesting: it trains models **sequentially**, "
    "where each new model specifically focuses on the mistakes the previous models "
    "made. It is less like asking a hundred people and more like having one student "
    "take a test, getting back the graded results, studying the questions they got "
    "wrong, taking the test again, and repeating until they ace it. This chapter "
    "covers gradient boosting, the learning rate, early stopping, and the "
    "hyperparameters that make or break a boosting model."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 48.1 Boosting Concepts ──────────────────────────────────────────────────
st.header("48.1  How Boosting Works")

concept_box(
    "Sequential Learning",
    "Boosting builds an <b>additive model</b>: start with a bad prediction, then "
    "ask 'where am I most wrong?' and train a small tree to fix those specific "
    "mistakes. Add that correction. Now ask again: 'where am I most wrong now?' "
    "Train another small tree. Repeat. Each weak learner (typically a shallow tree "
    "with just a few splits) is fitted to the <b>residuals</b> -- the errors of the "
    "current ensemble. The final prediction is the sum of all these corrections, "
    "each scaled down by a <b>learning rate</b> that controls how aggressive each "
    "step is."
)

formula_box(
    "Gradient Boosting Update",
    r"F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)",
    "F_m = ensemble after m iterations, eta = learning rate (how much trust to put in each new tree), h_m = weak learner fitted to the residuals. Small eta means small, careful steps."
)

st.markdown("""
**Key differences from Bagging:**
| | Bagging | Boosting |
|---|---|---|
| Training | Parallel (independent) | Sequential (each tree depends on previous) |
| Primary target | Reduce variance | Reduce bias |
| Overfitting risk | Low (more trees never hurts) | Higher (can overfit -- needs early stopping) |
| Tree depth | Deep, fully grown trees | Shallow stumps (2-5 levels) |
""")

# ── 48.2 Learning Rate & n_estimators ────────────────────────────────────────
st.header("48.2  Interactive: Learning Rate and Number of Estimators")

X_train, X_test, y_train, y_test, le, scaler = prepare_classification_data(
    fdf, FEATURE_COLS, target="city", test_size=0.2
)
city_labels = le.classes_

# Subsample for speed
sample_n = min(5000, len(X_train))
rng = np.random.RandomState(42)
idx = rng.choice(len(X_train), sample_n, replace=False)
X_tr_s = X_train.iloc[idx]
y_tr_s = y_train[idx]

col_lr, col_ne = st.columns(2)
with col_lr:
    learning_rate = st.select_slider(
        "Learning Rate (eta)",
        options=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
        value=0.1,
        key="boost_lr",
    )
with col_ne:
    n_estimators = st.slider("Number of Estimators", 10, 300, 100, 10, key="boost_n_est")

with st.spinner("Training Gradient Boosting..."):
    gb = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=3, random_state=42, validation_fraction=0.15,
        n_iter_no_change=20,
    )
    gb.fit(X_tr_s, y_tr_s)

gb_train_acc = accuracy_score(y_tr_s, gb.predict(X_tr_s))
gb_test_acc = accuracy_score(y_test, gb.predict(X_test))

c1, c2, c3 = st.columns(3)
c1.metric("Train Accuracy", f"{gb_train_acc:.4f}")
c2.metric("Test Accuracy", f"{gb_test_acc:.4f}")
c3.metric("Actual Estimators Used", f"{gb.n_estimators_}")

if gb.n_estimators_ < n_estimators:
    st.info(f"Early stopping kicked in at {gb.n_estimators_} estimators (you requested {n_estimators}). The model decided it was done learning before you told it to stop -- this is a good thing.")

# ── 48.3 Staged Predictions (Training Curve) ────────────────────────────────
st.header("48.3  Training Curve: Error vs Boosting Iterations")

# Train without early stopping to see full curve
gb_full = GradientBoostingClassifier(
    n_estimators=min(n_estimators, 200), learning_rate=learning_rate,
    max_depth=3, random_state=42,
)
gb_full.fit(X_tr_s, y_tr_s)

train_scores_staged = []
test_scores_staged = []
for i, (y_tr_pred, y_te_pred) in enumerate(
    zip(gb_full.staged_predict(X_tr_s), gb_full.staged_predict(X_test))
):
    train_scores_staged.append(accuracy_score(y_tr_s, y_tr_pred))
    test_scores_staged.append(accuracy_score(y_test, y_te_pred))

iters = list(range(1, len(train_scores_staged) + 1))

fig_staged = go.Figure()
fig_staged.add_trace(go.Scatter(
    x=iters, y=train_scores_staged,
    mode="lines", name="Train Accuracy",
    line=dict(color="#2A9D8F"),
))
fig_staged.add_trace(go.Scatter(
    x=iters, y=test_scores_staged,
    mode="lines", name="Test Accuracy",
    line=dict(color="#E63946"),
))
# Mark where test accuracy peaks
best_iter = np.argmax(test_scores_staged) + 1
fig_staged.add_vline(x=best_iter, line_dash="dash", line_color="gray",
                     annotation_text=f"Best iter: {best_iter}")
apply_common_layout(fig_staged, title=f"Boosting Curve (LR={learning_rate})", height=450)
fig_staged.update_layout(xaxis_title="Number of Estimators", yaxis_title="Accuracy")
st.plotly_chart(fig_staged, use_container_width=True)

insight_box(
    f"The best test accuracy ({max(test_scores_staged):.4f}) happens at iteration "
    f"**{best_iter}**. After that, the model keeps improving on training data (the "
    "green line keeps going up) while test accuracy stagnates or drops. This is "
    "boosting overfitting -- the model starts memorizing training noise instead of "
    "learning real patterns. Early stopping catches this automatically."
)

# ── 48.4 Effect of Learning Rate ─────────────────────────────────────────────
st.header("48.4  Learning Rate: Slow and Steady vs Fast and Reckless")

concept_box(
    "Learning Rate Tradeoff",
    "The learning rate is like the step size when walking downhill in fog. "
    "A <b>small learning rate</b> (e.g., 0.01) takes tiny, careful steps -- you "
    "will eventually reach the bottom but it takes forever. A <b>large learning "
    "rate</b> (e.g., 1.0) takes huge leaps -- you might overshoot the valley entirely "
    "and end up oscillating on the other side. The golden rule of boosting: "
    "<b>shrink the learning rate and grow the ensemble</b>. Slow and steady actually "
    "wins this race."
)

lr_values = [0.01, 0.05, 0.1, 0.3, 1.0]
lr_colors = ["#7209B7", "#FB8500", "#2A9D8F", "#264653", "#E63946"]

fig_lr = go.Figure()
for lr, color in zip(lr_values, lr_colors):
    gb_lr = GradientBoostingClassifier(
        n_estimators=150, learning_rate=lr, max_depth=3, random_state=42,
    )
    gb_lr.fit(X_tr_s, y_tr_s)
    test_sc = [accuracy_score(y_test, p) for p in gb_lr.staged_predict(X_test)]
    fig_lr.add_trace(go.Scatter(
        x=list(range(1, len(test_sc) + 1)), y=test_sc,
        mode="lines", name=f"LR={lr}",
        line=dict(color=color),
    ))
apply_common_layout(fig_lr, title="Test Accuracy vs Iterations at Different Learning Rates", height=450)
fig_lr.update_layout(xaxis_title="Number of Estimators", yaxis_title="Test Accuracy")
st.plotly_chart(fig_lr, use_container_width=True)

insight_box(
    "Lower learning rates reach higher peak accuracy but take more iterations to get "
    "there. Higher learning rates converge fast but plateau at a lower level -- or even "
    "start degrading as the model overfits. The practical advice: set the learning rate "
    "small (0.01-0.1), crank up the number of estimators, and let early stopping tell "
    "you when to stop. This almost always outperforms trying to guess the right number "
    "of iterations yourself."
)

# ── 48.5 XGBoost Comparison ─────────────────────────────────────────────────
st.header("48.5  XGBoost / Advanced Boosting")

try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False

if has_xgb:
    xgb = XGBClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=3, random_state=42, eval_metric="mlogloss",
        use_label_encoder=False, n_jobs=-1,
    )
    xgb.fit(X_tr_s, y_tr_s)
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
else:
    st.info("XGBoost not installed. Showing sklearn GradientBoosting results only.")
    xgb_acc = None

# Compare all methods
rf_comp = RandomForestClassifier(n_estimators=n_estimators, max_depth=10, random_state=42, n_jobs=-1)
rf_comp.fit(X_tr_s, y_tr_s)
rf_acc = accuracy_score(y_test, rf_comp.predict(X_test))

comparison_data = {
    "Model": ["Random Forest", "Gradient Boosting"],
    "Test Accuracy": [rf_acc, gb_test_acc],
}
if xgb_acc is not None:
    comparison_data["Model"].append("XGBoost")
    comparison_data["Test Accuracy"].append(xgb_acc)

comp_df = pd.DataFrame(comparison_data)
comp_df = comp_df.sort_values("Test Accuracy", ascending=False)

fig_comp = px.bar(comp_df, x="Model", y="Test Accuracy",
                  color="Model", color_discrete_sequence=["#E63946", "#2A9D8F", "#264653"],
                  title="Ensemble Method Comparison")
apply_common_layout(fig_comp, height=400)
st.plotly_chart(fig_comp, use_container_width=True)
st.dataframe(comp_df.style.format({"Test Accuracy": "{:.4f}"}), use_container_width=True, hide_index=True)

# ── 48.6 Hyperparameter Tuning Grid ─────────────────────────────────────────
st.header("48.6  Hyperparameter Tuning Grid Results")

concept_box(
    "Key Hyperparameters",
    "Boosting has several knobs, and they interact in non-obvious ways:<br>"
    "- <b>n_estimators</b>: how many boosting rounds. More is usually better, up to a point.<br>"
    "- <b>learning_rate</b>: step size shrinkage. Smaller = more conservative = usually better "
    "(but needs more rounds).<br>"
    "- <b>max_depth</b>: how deep each tree is. Deeper trees capture more complexity but risk "
    "overfitting. Usually 3-7 is the sweet spot.<br>"
    "- <b>subsample</b>: fraction of data used per tree. Less than 1.0 adds randomness "
    "(stochastic boosting), which can help generalization."
)

st.markdown("Here is a grid search over learning_rate and max_depth (with n_estimators=100) so you can see how these interact:")

grid_results = []
lr_grid = [0.01, 0.05, 0.1, 0.3]
depth_grid = [2, 3, 5, 7]

with st.spinner("Running hyperparameter grid..."):
    for lr in lr_grid:
        for depth in depth_grid:
            gb_grid = GradientBoostingClassifier(
                n_estimators=100, learning_rate=lr, max_depth=depth,
                random_state=42,
            )
            gb_grid.fit(X_tr_s, y_tr_s)
            acc = accuracy_score(y_test, gb_grid.predict(X_test))
            grid_results.append({"Learning Rate": lr, "Max Depth": depth, "Test Accuracy": acc})

grid_df = pd.DataFrame(grid_results)

# Pivot table for heatmap
pivot = grid_df.pivot(index="Max Depth", columns="Learning Rate", values="Test Accuracy")

fig_grid = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=[str(c) for c in pivot.columns],
    y=[str(r) for r in pivot.index],
    colorscale="RdYlGn",
    text=np.round(pivot.values, 4),
    texttemplate="%{text}",
))
apply_common_layout(fig_grid, title="Hyperparameter Grid: Test Accuracy", height=400)
fig_grid.update_layout(
    xaxis_title="Learning Rate",
    yaxis_title="Max Depth",
)
st.plotly_chart(fig_grid, use_container_width=True)

best_row = grid_df.loc[grid_df["Test Accuracy"].idxmax()]
st.success(
    f"Best configuration: **learning_rate={best_row['Learning Rate']}, "
    f"max_depth={int(best_row['Max Depth'])}** with test accuracy = {best_row['Test Accuracy']:.4f}"
)

# Sorted results table
st.dataframe(
    grid_df.sort_values("Test Accuracy", ascending=False)
    .style.format({"Test Accuracy": "{:.4f}"})
    .highlight_max(subset=["Test Accuracy"], color="#d4edda"),
    use_container_width=True, hide_index=True,
)

code_example("""from sklearn.ensemble import GradientBoostingClassifier

# Gradient Boosting with early stopping
gb = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=3,
    validation_fraction=0.15,
    n_iter_no_change=20,   # early stopping patience
    random_state=42,
)
gb.fit(X_train, y_train)
print(f"Stopped at {gb.n_estimators_} iterations")

# XGBoost
from xgboost import XGBClassifier
xgb = XGBClassifier(
    n_estimators=500, learning_rate=0.1, max_depth=3,
    early_stopping_rounds=20, eval_metric='mlogloss'
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "In boosting, each new tree is trained on:",
    [
        "A random bootstrap sample of the data",
        "The residuals (errors) of the current ensemble",
        "A completely new dataset",
        "The same data as the first tree",
    ],
    correct_idx=1,
    explanation="Each weak learner in gradient boosting is fitted to the gradient of the loss (which, for squared error, is just the residuals). It is literally learning from the ensemble's mistakes -- the parts where the current model is most wrong get the most attention from the next tree.",
    key="ch48_quiz1",
)

quiz(
    "A smaller learning rate in boosting means:",
    [
        "Fewer estimators are needed",
        "Each tree contributes less, requiring more trees for good performance",
        "The model trains faster",
        "The model will always underfit",
    ],
    correct_idx=1,
    explanation="A smaller learning rate means each tree makes a smaller correction -- like taking baby steps instead of leaps. You need more trees to reach the same performance, but the result usually generalizes better because you are less likely to overshoot.",
    key="ch48_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "Boosting trains weak learners **sequentially**, each one focusing on the mistakes the ensemble has made so far. It is the 'study what you got wrong' approach to machine learning.",
    "The **learning rate** controls how much trust to place in each new tree. Smaller is almost always better -- just pair it with more trees.",
    "**Early stopping** prevents overfitting by halting when validation performance plateaus. It is the boosting equivalent of knowing when to stop studying.",
    "Gradient boosting primarily reduces **bias** (unlike bagging, which reduces variance). This is why boosting often wins on complex tasks.",
    "Key hyperparameters: n_estimators, learning_rate, max_depth, subsample. They interact, so grid search or Bayesian optimization is your friend.",
    "XGBoost and LightGBM are optimized implementations that add regularization, better tree-building algorithms, and massive speed improvements over sklearn.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 47: Bagging",
    prev_page="47_Bagging.py",
    next_label="Ch 49: Stacking",
    next_page="49_Stacking.py",
)
