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
    "In the last chapter, bagging trained 100 trees independently and took a vote. "
    "Each tree saw a different bootstrap sample of weather data, but none of them "
    "knew what the other trees were doing. Boosting takes a fundamentally different "
    "approach: it trains trees **one at a time, in sequence**, where each new tree "
    "specifically focuses on the weather readings that the previous trees got wrong."
)
st.markdown(
    "**The task is the same**: predict which of 6 cities a weather reading came from, "
    "using temperature, humidity, wind speed, and pressure. But the strategy is different. "
    "Imagine a student taking a city-identification exam. After the first attempt, they "
    "get their graded test back: 'You confused Dallas with San Antonio 40 times, and "
    "you missed 20 NYC readings.' For the second attempt, they study *specifically* the "
    "Dallas-vs-San-Antonio distinction and the NYC misses. Third attempt, they focus on "
    "whatever they still get wrong. Each round targets the remaining weak spots."
)
st.markdown(
    "That is boosting. The first tree makes a rough prediction. The second tree is "
    "trained on the *errors* of the first. The third tree is trained on the errors "
    "that remain after combining the first two. And so on, each tree filling in the "
    "gaps left by its predecessors."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 48.1 Boosting Concepts ──────────────────────────────────────────────────
st.header("48.1  How Boosting Works")

concept_box(
    "Sequential Error Correction",
    "Let me walk through what happens step by step on our weather data:<br><br>"
    "<b>Round 1</b>: Train a small tree (maybe depth=3, only 3 questions). It learns "
    "the biggest patterns: 'cold readings are NYC, hot + humid readings are Houston.' "
    "But it confuses Dallas and San Antonio badly, because they need more nuance than "
    "3 questions can provide.<br><br>"
    "<b>Round 2</b>: Look at the Round 1 errors. Many are Dallas-San Antonio mix-ups. "
    "Train a new small tree, but <em>weighted toward those errors</em>. This tree might "
    "discover that San Antonio has slightly lower surface pressure on average.<br><br>"
    "<b>Round 3</b>: The remaining errors might be Austin vs Houston confusions during "
    "summer (both hot and humid). The next tree focuses on those.<br><br>"
    "Each tree is intentionally weak (shallow). The strength comes from the <em>sequence</em> "
    "of corrections. The final prediction is the sum of all these small corrections, each "
    "scaled down by a <b>learning rate</b> that controls how aggressively each step corrects."
)

formula_box(
    "Gradient Boosting Update",
    r"\underbrace{F_m(x)}_{\text{updated ensemble}} = \underbrace{F_{m-1}(x)}_{\text{previous prediction}} + \underbrace{\eta}_{\text{learning rate}} \cdot \underbrace{h_m(x)}_{\text{error-correcting tree}}",
    "F_m = the ensemble's prediction after m rounds. eta = learning rate (how much trust "
    "to put in each new tree -- typically 0.01 to 0.3). h_m = the weak learner trained on "
    "the residual errors. Small eta means 'take cautious baby steps'; large eta means "
    "'trust each new tree a lot.'"
)

st.markdown("""
**How does this differ from bagging?**

| | Bagging | Boosting |
|---|---|---|
| Training | Parallel -- trees are independent | Sequential -- each tree depends on the errors of the previous ones |
| Primary target | Reduce variance (stabilize noisy models) | Reduce bias (make simple models more powerful) |
| Overfitting risk | Low -- more trees never hurts | Higher -- too many rounds can memorize training noise |
| Tree depth | Deep, fully grown trees | Shallow stumps (2-5 levels) |
| Weather analogy | 100 forecasters work independently, then vote | One forecaster studies their mistakes after each attempt |
""")

# ── 48.2 Learning Rate & n_estimators ────────────────────────────────────────
st.header("48.2  Interactive: Learning Rate and Number of Estimators")

st.markdown(
    "The two most important knobs in boosting are the **learning rate** (how big each "
    "correction step is) and **n_estimators** (how many correction steps to take). "
    "They interact: a small learning rate needs more estimators to reach the same "
    "performance. Let us see this in action."
)

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

st.markdown(
    f"**Learning rate = {learning_rate}**: Each new tree's correction is multiplied by "
    f"{learning_rate} before being added to the ensemble. A value of 0.1 means 'only "
    f"trust each tree 10%.' This prevents any single tree from dominating the prediction.\n\n"
    f"**n_estimators = {n_estimators}**: The maximum number of correction rounds. More "
    f"rounds = more chances to fix remaining errors, but also more chances to memorize noise."
)

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
    st.info(
        f"Early stopping kicked in at {gb.n_estimators_} estimators (you requested "
        f"{n_estimators}). The model was monitoring a held-out validation set, and it "
        f"noticed that after round {gb.n_estimators_}, the validation accuracy stopped "
        f"improving. Additional rounds would only memorize training noise. This is the "
        f"boosting equivalent of knowing when to stop studying -- the last few repetitions "
        f"are not making you smarter, they are making you anxious."
    )

# ── 48.3 Staged Predictions (Training Curve) ────────────────────────────────
st.header("48.3  Training Curve: Watching Boosting Learn (and Overlearn)")

st.markdown(
    "Here is where boosting's Achilles heel shows up. Let us track both training and "
    "test accuracy at every boosting round and see where they diverge."
)

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
    f"**{best_iter}**. Up to that point, each new tree is learning real patterns: "
    f"'San Antonio's pressure is slightly lower than Dallas's,' 'LA readings rarely "
    f"have humidity above 80%.' After that point, the useful patterns have been "
    f"exhausted. Each new tree starts learning things like 'training reading #2,847 "
    f"has this exact combination of features, so it must be Austin' -- noise, not signal. "
    f"The green line (training accuracy) keeps climbing because memorizing noise helps on "
    f"the training set, but the red line (test accuracy) stagnates or drops. This is why "
    f"early stopping exists: it watches the test curve and stops at the peak."
)

# ── 48.4 Effect of Learning Rate ─────────────────────────────────────────────
st.header("48.4  Learning Rate: Baby Steps vs Giant Leaps")

st.markdown(
    "The learning rate controls how much each tree's correction counts. Let us see "
    "what happens when you change it."
)

concept_box(
    "The Learning Rate Tradeoff in Weather Terms",
    "Imagine you are trying to tune a thermostat to exactly 22.0 C.<br><br>"
    "- <b>Large learning rate (1.0)</b>: Each adjustment moves the dial by the full "
    "amount. You overshoot from 25 to 19, then back to 24, then 20... you oscillate "
    "wildly and might never settle.<br>"
    "- <b>Small learning rate (0.01)</b>: Each adjustment nudges the dial by 1% of the "
    "needed correction. 25.0 -> 24.97 -> 24.94 -> ... You will get there eventually, "
    "but it takes 500 nudges.<br>"
    "- <b>Medium learning rate (0.1)</b>: Each adjustment is 10% of the correction. "
    "25.0 -> 24.7 -> 24.4 -> ... Fast enough to converge in a reasonable time, slow "
    "enough to not overshoot.<br><br>"
    "The golden rule: <b>shrink the learning rate and increase the number of estimators.</b> "
    "Small, careful steps with many iterations almost always outperforms large, reckless "
    "leaps with few iterations."
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
    "Watch the curves carefully. LR=1.0 (red) shoots up fast but plateaus early and may "
    "even decline -- each tree overcorrects, and after a few dozen rounds, the model "
    "starts memorizing noise. LR=0.01 (purple) climbs slowly but steadily and has not "
    "peaked by iteration 150 -- it needs more rounds, but it will likely reach a higher "
    "ceiling. LR=0.1 (green) is the typical sweet spot: fast enough to converge in a "
    "reasonable time, slow enough to find a good solution. The practical recipe: set "
    "LR to 0.05 or 0.1, set n_estimators to 500 or 1,000, and let early stopping tell "
    "you when to stop."
)

# ── 48.5 XGBoost Comparison ─────────────────────────────────────────────────
st.header("48.5  Comparing Ensemble Methods on City Weather Data")

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

st.markdown(
    "On our weather data, the difference between Random Forest and Gradient Boosting "
    "may be small. This is common for tabular data with a moderate number of features. "
    "Where boosting really shines is on complex tasks where there are subtle, hard-to-find "
    "patterns -- like distinguishing Dallas from San Antonio using small differences in "
    "humidity-pressure combinations that a single-pass model might miss."
)

# ── 48.6 Hyperparameter Tuning Grid ─────────────────────────────────────────
st.header("48.6  Hyperparameter Tuning Grid Results")

st.markdown(
    "Boosting has several knobs, and they interact in non-obvious ways. The two most "
    "important are learning_rate and max_depth. Let us grid-search them on our city "
    "weather classification and see which combinations work best."
)

concept_box(
    "Key Hyperparameters for Our Weather Task",
    "- <b>n_estimators</b>: How many correction rounds. With learning_rate=0.1, you "
    "might need 100-200 rounds to classify 6 cities well. With learning_rate=0.01, you "
    "might need 500+.<br>"
    "- <b>learning_rate</b>: Step size per round. Smaller means more cautious corrections "
    "to the city predictions.<br>"
    "- <b>max_depth</b>: How many questions each small tree can ask. Depth=2 means each "
    "tree can only make 3 splits (e.g., 'Is temp > 25? If yes, is humidity > 60? If yes, "
    "is pressure > 1013?'). Depth=7 means 127 possible leaf nodes -- much more expressive "
    "but higher overfitting risk.<br>"
    "- <b>subsample</b>: Fraction of data used per tree (less than 1.0 adds randomness, "
    "like a mini-bagging within the boosting -- this is called stochastic gradient boosting)."
)

st.markdown(
    "Here is a grid search over learning_rate and max_depth (with n_estimators fixed "
    "at 100). The heatmap shows test accuracy for each combination -- greener is better:"
)

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

st.markdown(
    f"Notice the pattern in the heatmap. Very small learning rates (0.01) with only 100 "
    f"estimators tend to underperform -- they have not had enough rounds to converge. "
    f"Very deep trees (depth=7) with aggressive learning rates tend to overfit. The sweet "
    f"spot is typically a moderate learning rate (0.05-0.1) with moderate depth (3-5). "
    f"For our weather data, max_depth=3 often works well because the city-distinguishing "
    f"patterns are relatively simple: 3 questions like 'Is it cold? Is it humid? Is the "
    f"pressure high?' are often enough to narrow down the city."
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
        "A random bootstrap sample of weather readings (like bagging)",
        "The residuals (errors) of the current ensemble's city predictions",
        "A completely new dataset from different cities",
        "The same original weather data, unchanged",
    ],
    correct_idx=1,
    explanation=(
        "Each new tree in gradient boosting is fitted to the gradient of the loss -- "
        "which, intuitively, means it focuses on the weather readings where the current "
        "ensemble is most wrong. If the ensemble keeps confusing Dallas for San Antonio, "
        "the next tree concentrates its learning effort on those ambiguous Dallas/San "
        "Antonio readings, trying to find subtle pressure or humidity differences that "
        "could help separate them."
    ),
    key="ch48_quiz1",
)

quiz(
    "A smaller learning rate in boosting means:",
    [
        "Fewer estimators are needed to reach good accuracy",
        "Each tree contributes less, requiring more trees for good performance",
        "The model trains faster because less computation per round",
        "The model will always underfit, regardless of n_estimators",
    ],
    correct_idx=1,
    explanation=(
        "A learning rate of 0.01 means each tree's correction counts for only 1% of "
        "its full value. The model needs many rounds (perhaps 500+) to accumulate enough "
        "corrections to classify cities well. But these small, careful steps typically "
        "lead to a better final model than a few large, reckless leaps. The tradeoff: "
        "patience (more training time) for generalization (better test accuracy)."
    ),
    key="ch48_quiz2",
)

quiz(
    "Test accuracy peaks at iteration 80 but you trained for 200 iterations. What happened after iteration 80?",
    [
        "The model stopped learning anything new",
        "The model started memorizing training noise instead of real weather patterns",
        "The learning rate automatically decreased",
        "The model switched from learning bias to learning variance",
    ],
    correct_idx=1,
    explanation=(
        "After iteration 80, the useful patterns (NYC is cold, LA is dry, Texas cities "
        "are hot and humid) have been learned. Each additional tree starts fitting to "
        "noise -- 'this specific training reading with temperature 22.37 C and humidity "
        "68.2% is Dallas, not San Antonio.' These ultra-specific rules help on the "
        "training set but hurt on new data. That is why training accuracy keeps rising "
        "(memorization helps on training data) while test accuracy plateaus or drops. "
        "Early stopping catches this by monitoring a validation set and stopping when it "
        "plateaus."
    ),
    key="ch48_quiz3",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "Boosting trains weak learners **sequentially**, each one focusing on the weather readings the ensemble still gets wrong. It is the 'study what you got wrong on the exam' approach.",
    "The **learning rate** controls how much each tree's correction counts. Smaller is almost always better -- set it to 0.05-0.1 and pair it with more estimators.",
    "**Early stopping** monitors validation accuracy and halts when it stops improving. Without it, boosting will keep training past the peak and start memorizing noise.",
    "Boosting primarily reduces **bias** (unlike bagging, which reduces variance). It turns a collection of weak 3-question trees into a powerful ensemble that can distinguish 6 cities.",
    "The best hyperparameter combinations for our weather data tend to be moderate: learning_rate around 0.05-0.1, max_depth around 3-5. Extreme values in either direction hurt.",
    "XGBoost and LightGBM are optimized implementations that add regularization, better tree-building algorithms, and massive speed improvements -- but the core idea is the same sequential error correction.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 47: Bagging",
    prev_page="47_Bagging.py",
    next_label="Ch 49: Stacking",
    next_page="49_Stacking.py",
)
