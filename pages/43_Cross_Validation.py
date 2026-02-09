"""Chapter 43: Train-Test Split & Cross-Validation -- K-fold, stratified, time series CV, data leakage."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(43, "Train-Test Split & Cross-Validation", part="X")
st.markdown(
    "Here is a thing that happens embarrassingly often in machine learning: someone "
    "trains a model, evaluates it on the same data they trained it on, gets 99% "
    "accuracy, and declares victory. This is like studying for an exam by memorizing "
    "the answer key, then taking the same exam and being shocked -- *shocked* -- "
    "that you aced it. **Cross-validation** is the antidote. It gives us an honest "
    "estimate of how a model will perform on data it has never seen, and it turns "
    "out the difference matters a lot."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 43.1 Train-Test Split ────────────────────────────────────────────────────
st.header("43.1  The Train-Test Split")

concept_box(
    "Why Split the Data?",
    "The fundamental idea is almost insultingly simple: hold some data back. "
    "<b>Train</b> your model on one chunk and <b>evaluate</b> it on a chunk it has "
    "never seen. This simulates what will actually happen when you deploy the model "
    "in the real world, and prevents you from confusing memorization with genuine "
    "understanding. A model that has memorized its training data is like a student "
    "who can recite the textbook word-for-word but cannot solve a single novel problem."
)

formula_box(
    "Common Split Ratios",
    r"\underbrace{\text{Train}}_{\text{model learns from}} : \underbrace{\text{Test}}_{\text{unseen evaluation}} = 80\!:\!20 \quad\text{or}\quad 70\!:\!30",
    "The exact ratio depends on how much data you have. With a million rows, you can afford a tiny test set. With a thousand, you might want 30% held out. There is no magic number, but 80/20 is the default people reach for when they do not want to think too hard about it."
)

st.subheader("Interactive: Adjust the Split Ratio")
test_pct = st.slider("Test set percentage", 10, 50, 20, 5, key="split_pct")

le = LabelEncoder()
X_full = fdf[FEATURE_COLS].dropna()
y_full = le.fit_transform(fdf.loc[X_full.index, "city"])

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=test_pct / 100, random_state=42, stratify=y_full
)

col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", f"{len(X_full):,}")
col2.metric("Training Samples", f"{len(X_train):,}")
col3.metric("Test Samples", f"{len(X_test):,}")

rf_split = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_split.fit(X_train, y_train)
train_acc = accuracy_score(y_train, rf_split.predict(X_train))
test_acc = accuracy_score(y_test, rf_split.predict(X_test))

c1, c2 = st.columns(2)
c1.metric("Train Accuracy", f"{train_acc:.3f}")
c2.metric("Test Accuracy", f"{test_acc:.3f}", delta=f"{test_acc - train_acc:.3f}")

insight_box(
    "Notice the gap between training and test accuracy? That gap is the model's "
    "self-deception -- the degree to which it has confused memorization with learning. "
    "A large gap screams **overfitting**. Cross-validation, which we are about to "
    "explore, helps us detect this more reliably than any single train-test split can."
)

code_example("""from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
""")

# ── 43.2 K-Fold Cross-Validation ────────────────────────────────────────────
st.header("43.2  K-Fold Cross-Validation")

concept_box(
    "K-Fold CV",
    "Cross-validation is like studying for an exam by having a friend quiz you on "
    "random chapters, rather than just re-reading the whole book and telling yourself "
    "you understand it. The data is split into <b>K equally-sized folds</b>. The "
    "model is trained K times, each time using K-1 folds for training and the "
    "remaining fold as the quiz. The final score is the <b>mean across all K folds</b>. "
    "This way, every single data point gets a turn being the surprise quiz question."
)

formula_box(
    "CV Score",
    r"\underbrace{\text{CV Score}}_{\text{average performance}} = \frac{1}{\underbrace{K}_{\text{number of folds}}}\sum_{k=1}^{K} \underbrace{\text{Score}_k}_{\text{fold k accuracy}}",
    "The standard deviation across folds is just as interesting as the mean -- it tells you how sensitive the model is to which data it happens to train on. High std means your model is fickle; low std means it is reliable."
)

st.subheader("Interactive: Choose K for K-Fold CV")
k_folds = st.slider("Number of folds (K)", 2, 15, 5, key="k_folds")

# Subsample for speed
sample_size = min(5000, len(X_full))
idx_sample = np.random.RandomState(42).choice(len(X_full), sample_size, replace=False)
X_sample = X_full.iloc[idx_sample]
y_sample = y_full[idx_sample]

rf_cv = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)

with st.spinner(f"Running {k_folds}-fold cross-validation..."):
    scores = cross_val_score(rf_cv, X_sample, y_sample, cv=k_folds, scoring="accuracy")

# Fold results table
fold_df = pd.DataFrame({
    "Fold": [f"Fold {i+1}" for i in range(k_folds)],
    "Accuracy": scores,
})
fold_df.loc[len(fold_df)] = ["Mean +/- Std", scores.mean()]

col_a, col_b = st.columns([1, 2])
with col_a:
    st.dataframe(fold_df.style.format({"Accuracy": "{:.4f}"}), use_container_width=True, hide_index=True)
    st.write(f"**Mean:** {scores.mean():.4f} +/- {scores.std():.4f}")

with col_b:
    fig_folds = go.Figure()
    fig_folds.add_trace(go.Bar(
        x=[f"Fold {i+1}" for i in range(k_folds)],
        y=scores,
        marker_color="#2A9D8F",
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
    ))
    fig_folds.add_hline(y=scores.mean(), line_dash="dash", line_color="red",
                        annotation_text=f"Mean = {scores.mean():.4f}")
    apply_common_layout(fig_folds, title=f"{k_folds}-Fold CV Accuracy Scores", height=400)
    fig_folds.update_layout(yaxis_title="Accuracy", xaxis_title="Fold")
    st.plotly_chart(fig_folds, use_container_width=True)

# Variance across folds for different K
st.subheader("Score Variance Across Different K Values")
k_values = list(range(2, 16))
means_list = []
stds_list = []
for k in k_values:
    sc = cross_val_score(rf_cv, X_sample, y_sample, cv=k, scoring="accuracy")
    means_list.append(sc.mean())
    stds_list.append(sc.std())

var_df = pd.DataFrame({"K": k_values, "Mean Accuracy": means_list, "Std Dev": stds_list})
fig_var = go.Figure()
fig_var.add_trace(go.Scatter(
    x=var_df["K"], y=var_df["Mean Accuracy"],
    error_y=dict(type="data", array=var_df["Std Dev"]),
    mode="lines+markers", marker_color="#E63946", name="CV Accuracy",
))
apply_common_layout(fig_var, title="CV Score Mean +/- Std by K", height=400)
fig_var.update_layout(xaxis_title="K (number of folds)", yaxis_title="Accuracy")
st.plotly_chart(fig_var, use_container_width=True)

insight_box(
    "There is a satisfying tradeoff hiding here. Higher K means each fold has more "
    "training data, which reduces bias -- your estimate is closer to the truth. But "
    "the folds become more correlated with each other (they share more training data), "
    "and computation time scales linearly. K=5 or K=10 is the sweet spot for most people, "
    "which is why those are the defaults you see everywhere."
)

code_example("""from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print(f"Mean: {scores.mean():.4f} +/- {scores.std():.4f}")
""")

# ── 43.3 Stratified K-Fold ──────────────────────────────────────────────────
st.header("43.3  Stratified K-Fold")

concept_box(
    "Stratified Splits",
    "Wait, you might say, what if one class is rare? Imagine you have 100 samples: "
    "95 are Dallas and 5 are NYC. A random 5-fold split could easily put all 5 NYC "
    "samples in the same fold, leaving the other 4 folds with zero NYC examples. "
    "Your model would literally never learn to recognize NYC. <b>Stratified K-Fold</b> "
    "fixes this by preserving the class distribution in every fold, so each fold is a "
    "microcosm of the full dataset."
)

# Show class distribution in our dataset
city_counts = pd.Series(y_sample).value_counts().sort_index()
city_labels = le.inverse_transform(city_counts.index)
dist_df = pd.DataFrame({"City": city_labels, "Count": city_counts.values})

fig_dist = px.bar(dist_df, x="City", y="Count", color="City",
                  color_discrete_map=CITY_COLORS, title="Class Distribution in Sample")
apply_common_layout(fig_dist, height=350)
st.plotly_chart(fig_dist, use_container_width=True)

# Compare regular vs stratified
st.subheader("Regular KFold vs Stratified KFold")
kf_regular = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

regular_scores = cross_val_score(rf_cv, X_sample, y_sample, cv=kf_regular, scoring="accuracy")
strat_scores = cross_val_score(rf_cv, X_sample, y_sample, cv=skf, scoring="accuracy")

comp_df = pd.DataFrame({
    "Method": ["Regular KFold", "Stratified KFold"],
    "Mean Accuracy": [regular_scores.mean(), strat_scores.mean()],
    "Std Dev": [regular_scores.std(), strat_scores.std()],
})
st.dataframe(comp_df.style.format({"Mean Accuracy": "{:.4f}", "Std Dev": "{:.4f}"}),
             use_container_width=True, hide_index=True)

insight_box(
    "Stratified KFold usually gives **lower variance** across folds. This makes "
    "sense if you think about it: each fold has a representative sample of every class, "
    "so no fold is accidentally easier or harder than the others. It is basically "
    "free insurance, which is why scikit-learn uses it by default for classification."
)

# ── 43.4 Time Series Cross-Validation ───────────────────────────────────────
st.header("43.4  Time Series Cross-Validation (Walk-Forward)")

concept_box(
    "Time Series CV",
    "Here is where things get philosophically interesting. Standard K-Fold "
    "<b>shuffles</b> data randomly, which means your model might train on data from "
    "December to predict what happened in March. For time series, this is cheating -- "
    "you are literally using the future to predict the past. <b>Walk-forward CV</b> "
    "respects the arrow of time: always train on the past, test on the future. "
    "This is the only honest way to evaluate a time series model."
)

st.subheader("Walk-Forward Splits Visualization")
ts_city = st.selectbox("City for time series CV", CITY_LIST, key="ts_city")
city_ts = fdf[fdf["city"] == ts_city].sort_values("datetime").reset_index(drop=True)

n_ts_splits = st.slider("Number of time series splits", 3, 8, 5, key="ts_splits")
tscv = TimeSeriesSplit(n_splits=n_ts_splits)

fig_ts = go.Figure()
for i, (train_idx, test_idx) in enumerate(tscv.split(city_ts)):
    fig_ts.add_trace(go.Scatter(
        x=[train_idx[0], train_idx[-1]],
        y=[i + 1, i + 1],
        mode="lines",
        line=dict(color="#2A9D8F", width=12),
        name=f"Train {i+1}" if i == 0 else None,
        showlegend=(i == 0),
        legendgroup="train",
    ))
    fig_ts.add_trace(go.Scatter(
        x=[test_idx[0], test_idx[-1]],
        y=[i + 1, i + 1],
        mode="lines",
        line=dict(color="#E63946", width=12),
        name=f"Test {i+1}" if i == 0 else None,
        showlegend=(i == 0),
        legendgroup="test",
    ))
apply_common_layout(fig_ts, title="Walk-Forward CV Splits (sample index)", height=350)
fig_ts.update_layout(
    yaxis_title="Split", xaxis_title="Sample Index",
    yaxis=dict(dtick=1),
)
st.plotly_chart(fig_ts, use_container_width=True)

# Walk-forward temperature regression
st.subheader("Walk-Forward Regression: Predict Next-Hour Temperature")
features_ts = ["relative_humidity_pct", "wind_speed_kmh", "surface_pressure_hpa", "hour", "month"]
city_ts_clean = city_ts.dropna(subset=features_ts + ["temperature_c"])
X_ts = city_ts_clean[features_ts].values
y_ts = city_ts_clean["temperature_c"].values

tscv2 = TimeSeriesSplit(n_splits=n_ts_splits)
ts_rmses = []
for train_idx, test_idx in tscv2.split(X_ts):
    rf_ts = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
    rf_ts.fit(X_ts[train_idx], y_ts[train_idx])
    preds = rf_ts.predict(X_ts[test_idx])
    rmse = np.sqrt(mean_squared_error(y_ts[test_idx], preds))
    ts_rmses.append(rmse)

ts_result_df = pd.DataFrame({
    "Split": [f"Split {i+1}" for i in range(n_ts_splits)],
    "RMSE (deg C)": ts_rmses,
})
c1, c2 = st.columns([1, 2])
with c1:
    st.dataframe(ts_result_df.style.format({"RMSE (deg C)": "{:.3f}"}),
                 use_container_width=True, hide_index=True)
    st.write(f"**Mean RMSE:** {np.mean(ts_rmses):.3f} deg C")
with c2:
    fig_tsrmse = px.bar(ts_result_df, x="Split", y="RMSE (deg C)",
                        title="Walk-Forward RMSE by Split", color_discrete_sequence=["#264653"])
    fig_tsrmse.add_hline(y=np.mean(ts_rmses), line_dash="dash", line_color="red",
                         annotation_text=f"Mean = {np.mean(ts_rmses):.3f}")
    apply_common_layout(fig_tsrmse, height=400)
    st.plotly_chart(fig_tsrmse, use_container_width=True)

code_example("""from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
""")

# ── 43.5 Data Leakage Demonstration ─────────────────────────────────────────
st.header("43.5  Data Leakage Demonstration")

concept_box(
    "Data Leakage",
    "Data leakage is the number-one silent killer of ML projects. It happens when "
    "information from the <b>test set</b> (or from the future) sneaks into your "
    "training process. The insidious thing is that your metrics look fantastic "
    "during development -- the model appears to be a genius. Then you deploy it "
    "and everything falls apart. It is the machine learning equivalent of hiring "
    "someone who aced the interview because they had the answer sheet."
)

warning_box(
    "Leakage is insidious precisely because your metrics look great during development. "
    "You only discover the problem after deployment when performance collapses. By then, "
    "you have already sent the celebratory emails to your manager."
)

st.subheader("Demonstration: Scaling Before vs After Splitting")
st.markdown(
    "Here is a mistake that is so common it should have its own Wikipedia page: "
    "fitting `StandardScaler` on the **entire dataset** before splitting. The scaler "
    "then uses test-set statistics (mean, standard deviation) during training. "
    "The information leak is subtle but real."
)

# Correct: scale after split
X_tr, X_te, y_tr, y_te = train_test_split(X_sample, y_sample, test_size=0.2,
                                            random_state=42, stratify=y_sample)
scaler_correct = StandardScaler()
X_tr_correct = scaler_correct.fit_transform(X_tr)
X_te_correct = scaler_correct.transform(X_te)
rf_correct = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_correct.fit(X_tr_correct, y_tr)
acc_correct = accuracy_score(y_te, rf_correct.predict(X_te_correct))

# Leaky: scale before split
scaler_leaky = StandardScaler()
X_scaled_all = scaler_leaky.fit_transform(X_sample)
X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(X_scaled_all, y_sample, test_size=0.2,
                                                      random_state=42, stratify=y_sample)
rf_leaky = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_leaky.fit(X_tr_l, y_tr_l)
acc_leaky = accuracy_score(y_te_l, rf_leaky.predict(X_te_l))

st.subheader("Leakage: Adding a Future-Derived Feature")
st.markdown(
    "And here is a much more dramatic example of leakage: accidentally including a "
    "feature computed from **future** data. We add *next-hour temperature* as a "
    "predictor. Of course the model crushes it -- it is basically being handed the "
    "answer to the test, except the answer is one row down in the spreadsheet."
)

city_ts_leak = city_ts.dropna(subset=FEATURE_COLS).copy()
city_ts_leak["next_temp"] = city_ts_leak["temperature_c"].shift(-1)
city_ts_leak = city_ts_leak.dropna()

X_leak = city_ts_leak[FEATURE_COLS + ["next_temp"]].values
X_noleak = city_ts_leak[FEATURE_COLS].values
y_leak = city_ts_leak["temperature_c"].values

tscv_leak = TimeSeriesSplit(n_splits=3)
rmse_leak_list, rmse_noleak_list = [], []
for train_idx, test_idx in tscv_leak.split(X_leak):
    # With leakage
    rf_l = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
    rf_l.fit(X_leak[train_idx], y_leak[train_idx])
    rmse_leak_list.append(np.sqrt(mean_squared_error(y_leak[test_idx], rf_l.predict(X_leak[test_idx]))))
    # Without leakage
    rf_nl = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
    rf_nl.fit(X_noleak[train_idx], y_leak[train_idx])
    rmse_noleak_list.append(np.sqrt(mean_squared_error(y_leak[test_idx], rf_nl.predict(X_noleak[test_idx]))))

leak_comp = pd.DataFrame({
    "Scenario": ["Scale After Split (Correct)", "Scale Before Split (Leaky)",
                  "Without Future Feature", "With Future Feature (Leaked)"],
    "Metric": [f"Accuracy: {acc_correct:.4f}", f"Accuracy: {acc_leaky:.4f}",
               f"RMSE: {np.mean(rmse_noleak_list):.3f} C", f"RMSE: {np.mean(rmse_leak_list):.3f} C"],
    "Status": ["Correct", "Leaky (subtle)", "Correct", "Leaky (severe)"],
})
st.dataframe(leak_comp, use_container_width=True, hide_index=True)

fig_leak = go.Figure()
fig_leak.add_trace(go.Bar(name="No Leakage", x=["Scaling", "Future Feature"],
                          y=[acc_correct, np.mean(rmse_noleak_list)],
                          marker_color="#2A9D8F"))
fig_leak.add_trace(go.Bar(name="With Leakage", x=["Scaling", "Future Feature"],
                          y=[acc_leaky, np.mean(rmse_leak_list)],
                          marker_color="#E63946"))
apply_common_layout(fig_leak, title="Data Leakage Impact Comparison", height=400)
fig_leak.update_layout(barmode="group", yaxis_title="Metric Value")
st.plotly_chart(fig_leak, use_container_width=True)

warning_box(
    "Look at that future-feature leakage -- the model looks dramatically better. "
    "But in production, next-hour temperature would not be available (because the future "
    "has not happened yet, which is a surprisingly common thing to forget when writing "
    "feature pipelines at 2am). The deployed model would fail spectacularly."
)

code_example("""# WRONG -- leakage!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # uses ALL data
X_train, X_test = train_test_split(X_scaled, ...)

# CORRECT -- no leakage
X_train, X_test = train_test_split(X, ...)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit on train only
X_test = scaler.transform(X_test)          # transform test
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "In time series cross-validation, why can't we use standard K-Fold?",
    [
        "K-Fold is too slow for time series data",
        "K-Fold shuffles data, leaking future information into training",
        "K-Fold only works for classification, not regression",
        "Time series data has too many features for K-Fold",
    ],
    correct_idx=1,
    explanation="Standard K-Fold shuffles observations, so future data points end up in the training set. This violates temporal causality -- your model would literally be using next week's weather to predict this week's, which is not a skill it will have in production.",
    key="ch43_quiz1",
)

quiz(
    "What is data leakage?",
    [
        "When the model underfits the training data",
        "When training data is too small",
        "When test-set or future information influences training",
        "When you use too many folds in CV",
    ],
    correct_idx=2,
    explanation="Data leakage means information that should be unknown during training (test-set statistics, future values, the answers to the quiz) contaminates the model. Your metrics lie to you, and you only find out the hard way.",
    key="ch43_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "A single train-test split gives a noisy estimate -- it is one roll of the dice. Cross-validation averages over multiple splits to get a much more reliable picture.",
    "K-Fold CV trains the model K times, and the mean and standard deviation of scores tell you both *how good* and *how stable* your model is.",
    "Use **Stratified K-Fold** when class labels are imbalanced -- it is basically free insurance that every fold looks like the whole dataset.",
    "For time series, use **walk-forward (TimeSeriesSplit)** CV. Shuffling temporal data is the statistical equivalent of reading tomorrow's newspaper today.",
    "Data leakage inflates metrics but destroys real-world performance. Always split *before* preprocessing. Be paranoid about this.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 42",
    prev_page="42_Feature_Selection.py",
    next_label="Ch 44: Bias-Variance Tradeoff",
    next_page="44_Bias_Variance_Tradeoff.py",
)
