"""Chapter 45: ROC/AUC & Confusion Matrix -- Precision, recall, F1, threshold tuning."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve,
)

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.ml_helpers import (
    prepare_classification_data, classification_metrics, plot_confusion_matrix,
)
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(45, "ROC/AUC & Confusion Matrix", part="X")
st.markdown(
    "Accuracy is the metric everyone reaches for first, and it is also the metric "
    "most likely to lie to you. If 95% of your emails are not spam, a model that "
    "always predicts 'not spam' gets 95% accuracy and catches exactly zero spam. "
    "Congratulations, you have built the world's most confident do-nothing. This "
    "chapter introduces the tools that actually tell you what is going on: the "
    "**confusion matrix**, **precision**, **recall**, **F1-score**, **ROC curves**, "
    "and **AUC** -- the essential diagnostic toolkit for anyone building classifiers."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 45.1 Confusion Matrix Concepts ──────────────────────────────────────────
st.header("45.1  The Confusion Matrix")

concept_box(
    "Confusion Matrix",
    "A confusion matrix is, delightfully, named after the thing it diagnoses. "
    "It is an N x N table (for N classes) where entry (i, j) tells you how many "
    "samples of true class i were predicted as class j. The diagonal is where the "
    "model got things right; everything off the diagonal is a mistake. You can "
    "learn more from staring at a confusion matrix for five minutes than from any "
    "single number."
)

formula_box(
    "Key Metrics from the Confusion Matrix",
    r"\text{Precision} = \frac{TP}{TP + FP}, \quad "
    r"\text{Recall} = \frac{TP}{TP + FN}, \quad "
    r"F_1 = 2 \cdot \frac{P \cdot R}{P + R}",
    "TP = True Positive, FP = False Positive, FN = False Negative. Precision asks 'of things I called positive, how many actually were?' Recall asks 'of things that actually were positive, how many did I find?'"
)

# ── 45.2 6-City Confusion Matrix ────────────────────────────────────────────
st.header("45.2  6-City Weather Classification Confusion Matrix")

X_train, X_test, y_train, y_test, le, scaler = prepare_classification_data(
    fdf, FEATURE_COLS, target="city", test_size=0.2
)

rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

city_labels = le.classes_
metrics = classification_metrics(y_test, y_pred, labels=city_labels)

st.subheader("Confusion Matrix")
fig_cm = plot_confusion_matrix(metrics["confusion_matrix"], city_labels)
st.plotly_chart(fig_cm, use_container_width=True)

st.subheader("Per-City Metrics")
report_df = pd.DataFrame(metrics["report"]).T
report_df = report_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
report_df = report_df.round(3)
st.dataframe(report_df, use_container_width=True)

# Highlight Dallas-San Antonio confusion
cm = metrics["confusion_matrix"]
dallas_idx = list(city_labels).index("Dallas") if "Dallas" in city_labels else None
sa_idx = list(city_labels).index("San Antonio") if "San Antonio" in city_labels else None

if dallas_idx is not None and sa_idx is not None:
    d_as_sa = cm[dallas_idx, sa_idx]
    sa_as_d = cm[sa_idx, dallas_idx]
    st.subheader("Dallas vs San Antonio Confusion")
    st.markdown(
        f"Here is a fun one: the model keeps confusing Dallas and San Antonio. "
        f"And honestly, can you blame it?\n"
        f"- **Dallas predicted as San Antonio:** {d_as_sa} times\n"
        f"- **San Antonio predicted as Dallas:** {sa_as_d} times\n\n"
        f"These are both Texas cities separated by about 270 miles. The model is "
        f"essentially being asked to distinguish between 'Texas weather' and "
        f"'slightly different Texas weather.'"
    )
    insight_box(
        "Dallas and San Antonio share a similar climate zone -- hot summers, mild "
        "winters, and humidity patterns that overlap significantly. The model's "
        "confusion here is not a bug, it is a reflection of meteorological reality. "
        "Geography is the strongest predictor of weather, and these cities just are "
        "not that geographically different."
    )

code_example("""from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=city_names)
print(report)
""")

# ── 45.3 Precision, Recall, F1 Deep Dive ────────────────────────────────────
st.header("45.3  Precision, Recall, and F1-Score")

concept_box(
    "When Accuracy Is Not Enough",
    "Here is the scenario where accuracy falls apart: imbalanced classes. Imagine a "
    "disease test where only 1% of patients are sick. A model that always says "
    "'healthy' gets 99% accuracy and never catches a single sick person. "
    "<b>Precision</b> asks 'of those I flagged as sick, how many actually are?' "
    "(important if false alarms are costly). <b>Recall</b> asks 'of those who are "
    "actually sick, how many did I find?' (important if missing a case is catastrophic). "
    "They are two different kinds of being wrong, and which one you care about "
    "depends entirely on the cost structure of your problem."
)

# Bar chart of per-city precision, recall, F1
metric_data = []
for city in city_labels:
    row = metrics["report"][city]
    metric_data.append({"City": city, "Precision": row["precision"],
                        "Recall": row["recall"], "F1-Score": row["f1-score"]})
metric_df = pd.DataFrame(metric_data)

fig_metrics = go.Figure()
for metric_name, color in [("Precision", "#2A9D8F"), ("Recall", "#E63946"), ("F1-Score", "#264653")]:
    fig_metrics.add_trace(go.Bar(
        x=metric_df["City"], y=metric_df[metric_name],
        name=metric_name, marker_color=color,
    ))
apply_common_layout(fig_metrics, title="Precision, Recall, F1 by City", height=450)
fig_metrics.update_layout(barmode="group", yaxis_title="Score", xaxis_title="City")
st.plotly_chart(fig_metrics, use_container_width=True)

# ── 45.4 Binary Classification Threshold Tuning ─────────────────────────────
st.header("45.4  Interactive: Binary Threshold Tuning")

concept_box(
    "Threshold Tuning",
    "Most classifiers do not actually output 'yes' or 'no' -- they output a "
    "probability. We then pick a <b>threshold</b> (usually 0.5) and say 'above the "
    "threshold = positive.' But who says 0.5 is the right number? Lowering the "
    "threshold catches more true positives (higher recall) but also lets in more "
    "false alarms (lower precision). Raising it does the opposite. The right "
    "threshold depends entirely on what errors cost you. A cancer screening test "
    "and a spam filter want very different thresholds."
)

# Binary: one city vs rest
target_city = st.selectbox("Target city (positive class)", CITY_LIST, index=0, key="roc_city")
y_binary = (fdf.loc[X_train.index.tolist() + X_test.index.tolist(), "city"] == target_city).astype(int)
y_train_bin = y_binary.loc[X_train.index].values
y_test_bin = y_binary.loc[X_test.index].values

lr = LogisticRegression(max_iter=1000, random_state=42)
scaler_bin = StandardScaler()
X_tr_s = scaler_bin.fit_transform(X_train)
X_te_s = scaler_bin.transform(X_test)
lr.fit(X_tr_s, y_train_bin)

y_proba = lr.predict_proba(X_te_s)[:, 1]

threshold = st.slider("Classification Threshold", 0.01, 0.99, 0.50, 0.01, key="thresh_slider")

y_pred_thresh = (y_proba >= threshold).astype(int)
tp = ((y_pred_thresh == 1) & (y_test_bin == 1)).sum()
fp = ((y_pred_thresh == 1) & (y_test_bin == 0)).sum()
fn = ((y_pred_thresh == 0) & (y_test_bin == 1)).sum()
tn = ((y_pred_thresh == 0) & (y_test_bin == 0)).sum()

prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
acc_t = (tp + tn) / len(y_test_bin)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Precision", f"{prec:.3f}")
col2.metric("Recall", f"{rec:.3f}")
col3.metric("F1-Score", f"{f1:.3f}")
col4.metric("Accuracy", f"{acc_t:.3f}")

# Confusion matrix at current threshold
cm_bin = np.array([[tn, fp], [fn, tp]])
fig_cm_bin = go.Figure(data=go.Heatmap(
    z=cm_bin, x=["Not " + target_city, target_city],
    y=["Not " + target_city, target_city],
    colorscale="Blues", text=cm_bin, texttemplate="%{text}",
))
fig_cm_bin.update_layout(
    xaxis_title="Predicted", yaxis_title="Actual",
    title=f"Confusion Matrix (threshold={threshold:.2f})",
    height=400, template="plotly_white",
)
st.plotly_chart(fig_cm_bin, use_container_width=True)

# Precision-Recall vs Threshold
thresholds_range = np.linspace(0.01, 0.99, 50)
precs_list, recs_list, f1s_list = [], [], []
for t in thresholds_range:
    yp = (y_proba >= t).astype(int)
    p = precision_score(y_test_bin, yp, zero_division=0)
    r = recall_score(y_test_bin, yp, zero_division=0)
    f = f1_score(y_test_bin, yp, zero_division=0)
    precs_list.append(p)
    recs_list.append(r)
    f1s_list.append(f)

fig_thresh = go.Figure()
fig_thresh.add_trace(go.Scatter(x=thresholds_range, y=precs_list,
                                 mode="lines", name="Precision", line=dict(color="#2A9D8F")))
fig_thresh.add_trace(go.Scatter(x=thresholds_range, y=recs_list,
                                 mode="lines", name="Recall", line=dict(color="#E63946")))
fig_thresh.add_trace(go.Scatter(x=thresholds_range, y=f1s_list,
                                 mode="lines", name="F1-Score", line=dict(color="#264653")))
fig_thresh.add_vline(x=threshold, line_dash="dash", line_color="gray",
                     annotation_text=f"Current: {threshold:.2f}")
apply_common_layout(fig_thresh, title="Precision / Recall / F1 vs Threshold", height=400)
fig_thresh.update_layout(xaxis_title="Threshold", yaxis_title="Score")
st.plotly_chart(fig_thresh, use_container_width=True)

insight_box(
    "There is no free lunch here: as you drag the threshold lower, recall goes up "
    "(you catch more real positives) but precision goes down (you also cry wolf more "
    "often). The right balance depends on the **cost of each type of error** in your "
    "specific application. Missing a tornado warning? High cost -- favor recall. "
    "Sending a false 'pack an umbrella' notification? Low cost -- maybe optimize F1."
)

# ── 45.5 ROC Curve with AUC ─────────────────────────────────────────────────
st.header("45.5  ROC Curve and AUC")

concept_box(
    "ROC (Receiver Operating Characteristic)",
    "The ROC curve is one of those ideas that is simultaneously simple and profound. "
    "It plots <b>True Positive Rate</b> (recall) against <b>False Positive Rate</b> "
    "at every possible threshold, sweeping from 'predict everyone as negative' to "
    "'predict everyone as positive.' <b>AUC</b> (Area Under the Curve) compresses "
    "this into a single number: 0.5 means your model is no better than flipping a "
    "coin, 1.0 means it has achieved perfection. Most real models land somewhere "
    "in between, ideally much closer to 1."
)

formula_box(
    "TPR and FPR",
    r"\text{TPR} = \frac{TP}{TP+FN}, \qquad \text{FPR} = \frac{FP}{FP+TN}",
    "TPR is just recall by another name. FPR is the fraction of actual negatives that you incorrectly flagged as positive -- the false alarm rate."
)

fpr, tpr, roc_thresholds = roc_curve(y_test_bin, y_proba)
roc_auc = auc(fpr, tpr)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=fpr, y=tpr, mode="lines",
    name=f"ROC (AUC = {roc_auc:.3f})",
    line=dict(color="#E63946", width=2),
))
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines",
    name="Random (AUC = 0.5)",
    line=dict(color="gray", dash="dash"),
))
# Mark current threshold on ROC
idx_closest = np.argmin(np.abs(roc_thresholds - threshold))
fig_roc.add_trace(go.Scatter(
    x=[fpr[idx_closest]], y=[tpr[idx_closest]],
    mode="markers", marker=dict(size=12, color="#264653", symbol="star"),
    name=f"Threshold={threshold:.2f}",
))
apply_common_layout(fig_roc, title=f"ROC Curve: {target_city} vs Rest", height=450)
fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
st.plotly_chart(fig_roc, use_container_width=True)

st.metric("AUC Score", f"{roc_auc:.3f}")

# Multi-class ROC (one-vs-rest)
st.subheader("Multi-Class ROC (One-vs-Rest)")

rf_proba = rf.predict_proba(X_test)
y_test_bin_all = label_binarize(y_test, classes=list(range(len(city_labels))))

fig_multi_roc = go.Figure()
city_colors_list = [CITY_COLORS.get(c, "#333") for c in city_labels]
for i, (city, color) in enumerate(zip(city_labels, city_colors_list)):
    fpr_i, tpr_i, _ = roc_curve(y_test_bin_all[:, i], rf_proba[:, i])
    auc_i = auc(fpr_i, tpr_i)
    fig_multi_roc.add_trace(go.Scatter(
        x=fpr_i, y=tpr_i, mode="lines",
        name=f"{city} (AUC={auc_i:.3f})",
        line=dict(color=color),
    ))
fig_multi_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines",
    name="Random", line=dict(color="gray", dash="dash"),
))
apply_common_layout(fig_multi_roc, title="Multi-Class ROC Curves (One-vs-Rest)", height=500)
fig_multi_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
st.plotly_chart(fig_multi_roc, use_container_width=True)

insight_box(
    "Cities with unique climates (NYC with its cold winters, LA with its coastal "
    "Mediterranean vibes) tend to have higher AUC because the model can latch onto "
    "distinctive features. Dallas and San Antonio, which are basically weather "
    "siblings, get lower AUC because the model cannot reliably tell them apart. "
    "The ROC curve is basically a report card for how distinguishable each city is."
)

code_example("""from sklearn.metrics import roc_curve, auc

# Binary ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Multi-class: use label_binarize + one-vs-rest
from sklearn.preprocessing import label_binarize
y_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5])
for i in range(n_classes):
    fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_proba[:, i])
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "If AUC = 0.5, the model is performing:",
    [
        "Perfectly",
        "No better than random guessing",
        "With high precision",
        "With high recall",
    ],
    correct_idx=1,
    explanation="AUC of 0.5 means the ROC curve is the diagonal line -- the model has exactly zero discriminative ability. You would do equally well by flipping a coin. This is the ML equivalent of a participation trophy.",
    key="ch45_quiz1",
)

quiz(
    "Lowering the classification threshold will generally:",
    [
        "Increase precision, decrease recall",
        "Increase recall, decrease precision",
        "Increase both precision and recall",
        "Have no effect on predictions",
    ],
    correct_idx=1,
    explanation="A lower threshold means you say 'yes' to more things. You catch more true positives (higher recall) but you also wave through more false positives (lower precision). It is the 'better safe than sorry' setting.",
    key="ch45_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "The **confusion matrix** shows exactly where your model is confused -- and the pattern of its mistakes is often more informative than any single score.",
    "**Precision** answers 'when I say yes, am I right?' **Recall** answers 'of all the actual yeses, did I find them?' These are fundamentally different questions.",
    "**F1-Score** is the harmonic mean of precision and recall -- a compromise metric for when you cannot decide which error is worse.",
    "The **ROC curve** (TPR vs FPR) and its **AUC** give you a threshold-independent measure of how well your classifier separates classes.",
    "Dallas and San Antonio are frequently confused because they are meteorological near-twins. This is the model telling you something true about the world.",
    "**Threshold tuning** lets you trade precision for recall. The optimal threshold depends on the real-world costs of false positives vs false negatives.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 44: Bias-Variance Tradeoff",
    prev_page="44_Bias_Variance_Tradeoff.py",
    next_label="Ch 46: Regression Metrics",
    next_page="46_Regression_Metrics.py",
)
