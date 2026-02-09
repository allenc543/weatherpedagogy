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
    "Let me set up the problem we are actually solving, because these metrics only "
    "make sense in context."
)
st.markdown(
    "**The task**: You have a single weather reading -- temperature, humidity, wind "
    "speed, and surface pressure -- and you need to guess which of 6 cities it came "
    "from (Dallas, San Antonio, Houston, Austin, NYC, or Los Angeles). A random "
    "forest classifier looks at the numbers and says, 'I think this reading is from "
    "Houston.' How do you know if the model is any good at this?"
)
st.markdown(
    "Your first instinct is accuracy: 'What fraction of readings did it label "
    "correctly?' But accuracy has a dirty secret. Our dataset has 6 cities with "
    "roughly equal representation, so a model that just always guesses the most "
    "common city gets about 17% accuracy -- not great. But imagine a different "
    "scenario: if 90% of the readings were from Dallas, a model that always says "
    "'Dallas' gets 90% accuracy and has learned absolutely nothing. This chapter "
    "introduces the tools that tell you what is *actually* going on: the "
    "**confusion matrix**, **precision**, **recall**, **F1-score**, **ROC curves**, "
    "and **AUC**."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 45.1 Confusion Matrix Concepts ──────────────────────────────────────────
st.header("45.1  The Confusion Matrix")

st.markdown(
    "Before I define anything formally, let me show you the idea. The model looks "
    "at 21,000 test weather readings and predicts a city for each one. We know the "
    "true city. So for each pair (true city, predicted city) we can count how many "
    "times it happened. Lay that out in a grid and you get the confusion matrix."
)

concept_box(
    "Confusion Matrix",
    "Think of it this way: every row of the matrix is a city that weather readings "
    "actually came from. Every column is what the model predicted. The diagonal -- "
    "where the true city matches the predicted city -- is where the model got it "
    "right. Everything off the diagonal is a mistake, and the pattern of those "
    "mistakes tells you something specific. If the cell at (Dallas row, San Antonio "
    "column) has a big number, that means the model keeps looking at Dallas weather "
    "and saying 'San Antonio.' You can learn more from staring at this grid for five "
    "minutes than from any single accuracy number."
)

formula_box(
    "Key Metrics from the Confusion Matrix",
    r"\underbrace{\text{Precision}}_{\text{correct when predicted}} = \frac{\underbrace{TP}_{\text{true positives}}}{\underbrace{TP + FP}_{\text{all predicted positive}}}, \quad "
    r"\underbrace{\text{Recall}}_{\text{found from actual}} = \frac{\underbrace{TP}_{\text{true positives}}}{\underbrace{TP + FN}_{\text{all actual positive}}}, \quad "
    r"\underbrace{F_1}_{\text{balanced score}} = 2 \cdot \frac{\underbrace{P}_{\text{precision}} \cdot \underbrace{R}_{\text{recall}}}{P + R}",
    "TP = True Positive, FP = False Positive, FN = False Negative. For our city "
    "classification: if we focus on Houston, TP = readings correctly identified as "
    "Houston, FP = readings from other cities that the model wrongly called Houston, "
    "FN = actual Houston readings the model assigned to some other city."
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
st.markdown(
    "Each cell shows how many weather readings from the city on the left (true) "
    "were predicted as the city on top (predicted). The diagonal cells are correct "
    "predictions. Off-diagonal cells are mistakes -- and the *size* of each mistake "
    "tells you exactly which cities the model confuses."
)
fig_cm = plot_confusion_matrix(metrics["confusion_matrix"], city_labels)
st.plotly_chart(fig_cm, use_container_width=True)

st.subheader("Per-City Metrics")
st.markdown(
    "For each city, we can compute three numbers. **Precision** answers: of all the "
    "times the model said 'this reading is from Houston,' how many were actually from "
    "Houston? **Recall** answers: of all the readings that truly were from Houston, "
    "how many did the model correctly identify? **F1** is their harmonic mean -- a "
    "single compromise number."
)
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
    st.subheader("Dallas vs San Antonio: The Model's Hardest Puzzle")
    st.markdown(
        f"Look at the off-diagonal cells for Dallas and San Antonio. The model keeps "
        f"mixing them up, and honestly, so would you if you were only looking at "
        f"temperature, humidity, wind, and pressure:\n"
        f"- **Dallas readings predicted as San Antonio:** {d_as_sa} times\n"
        f"- **San Antonio readings predicted as Dallas:** {sa_as_d} times\n\n"
        f"These cities are about 270 miles apart in central Texas. In July, Dallas "
        f"averages around 35-36 degrees C; San Antonio averages around 34-35 degrees C. "
        f"Their humidity patterns overlap heavily. Their surface pressures are similar. "
        f"The model is being asked to distinguish between 'Texas summer' and 'slightly "
        f"different Texas summer' using four numbers, and it can't reliably do it."
    )
    insight_box(
        "This confusion is not a bug -- it is the model telling you something true "
        "about the world. Dallas and San Antonio share a climate zone (humid subtropical). "
        "Their temperature distributions overlap significantly, their humidity ranges are "
        "nearly identical, and without geographic features like latitude or elevation as "
        "inputs, the model simply does not have enough information to reliably separate "
        "them. If you wanted to fix this, you would need to add features that actually "
        "differ between these cities -- perhaps UV index, precipitation patterns, or "
        "time-of-day temperature swings."
    )

code_example("""from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=city_names)
print(report)
""")

# ── 45.3 Precision, Recall, F1 Deep Dive ────────────────────────────────────
st.header("45.3  Precision, Recall, and F1-Score")

st.markdown(
    "Let me make the distinction between precision and recall extremely concrete "
    "with our weather data, because this is one of those things that sounds abstract "
    "until you see it in action."
)

concept_box(
    "Precision vs Recall: Two Different Questions About Houston",
    "Suppose the model says 'Houston' 1,000 times across all test readings. Of those "
    "1,000 predictions, 800 actually came from Houston and 200 came from other cities "
    "(maybe 120 from Dallas and 80 from San Antonio -- cities with similar weather). "
    "The precision for Houston is 800/1000 = 0.80. In plain English: <b>when the model "
    "says Houston, it's right 80% of the time.</b><br><br>"
    "Now flip the question. There are actually 900 Houston readings in the test set. "
    "The model correctly identified 800 of them but missed 100 (calling them Dallas or "
    "Austin). The recall for Houston is 800/900 = 0.89. In plain English: <b>of all the "
    "readings that truly came from Houston, the model found 89% of them.</b><br><br>"
    "These measure different kinds of mistakes. Low precision means the model is trigger-happy -- "
    "it keeps saying 'Houston!' when it shouldn't. Low recall means the model is gun-shy -- "
    "it fails to recognize Houston readings when it should."
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

st.markdown(
    "Notice which cities have the highest scores and which have the lowest. Cities "
    "with distinctive weather signatures -- NYC with its cold winters, LA with its "
    "dry Mediterranean climate -- tend to score high on all three metrics. The Texas "
    "cities, which share similar climate patterns, tend to score lower because the "
    "model confuses them with each other."
)

# ── 45.4 Binary Classification Threshold Tuning ─────────────────────────────
st.header("45.4  Interactive: Binary Threshold Tuning")

st.markdown(
    "Now I want to simplify the problem to show you something important. Instead of "
    "'which of 6 cities?' let us ask a yes/no question: 'Is this weather reading "
    "from a specific city, or not?' This turns our 6-class problem into a binary one, "
    "and binary classification is where threshold tuning really shines."
)

concept_box(
    "What Is a Threshold, and Why Can You Move It?",
    "The logistic regression model does not actually say 'yes' or 'no.' It says "
    "something like 'I am 73% confident this reading is from Dallas.' We then apply "
    "a <b>threshold</b> -- usually 0.5 -- and say 'above 0.5 means yes, below means no.' "
    "But who decided 0.5 was the right cutoff?<br><br>"
    "Imagine you are building a weather alert system that detects whether conditions "
    "match a dangerous city's profile (say, for routing emergency resources). Missing "
    "a match could be costly. In that case, you might lower the threshold to 0.3: "
    "'If the model is even 30% sure, flag it.' You will catch more true matches (higher "
    "recall) but also get more false alarms (lower precision). Raising the threshold to "
    "0.7 does the opposite: fewer false alarms, but you miss more real matches."
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

st.markdown(
    f"The model outputs a probability for each test reading: 'how confident am I that "
    f"this reading is from {target_city}?' Drag the slider to change the threshold and "
    f"watch how precision, recall, and the confusion matrix shift in real time."
)

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
    f"Try dragging the threshold from 0.1 to 0.9 and watch what happens. At low "
    f"thresholds, the model says 'yes, this is {target_city}' to almost everything -- "
    f"recall is high (it catches nearly every real {target_city} reading) but precision "
    f"tanks (it is also flagging readings from Dallas, Houston, and everywhere else). "
    f"At high thresholds, the model becomes very picky -- precision is high (when it "
    f"says {target_city}, it is almost always right) but recall drops (it misses many "
    f"real {target_city} readings because it was not confident enough). The F1 curve "
    f"peaks somewhere in the middle, where the two forces balance."
)

# ── 45.5 ROC Curve with AUC ─────────────────────────────────────────────────
st.header("45.5  ROC Curve and AUC")

st.markdown(
    "The threshold slider above showed you one specific tradeoff at one specific "
    "threshold. But what if you want to evaluate the model's *overall* ability to "
    "separate one city from the rest, across all possible thresholds at once? "
    "That is exactly what the ROC curve does."
)

concept_box(
    "ROC (Receiver Operating Characteristic)",
    "Imagine sweeping the threshold from 1.0 (predict nobody as the target city) "
    "down to 0.0 (predict everybody as the target city). At each threshold, you "
    "compute two numbers:<br><br>"
    "- <b>True Positive Rate (TPR)</b>: of all readings actually from the target "
    "city, what fraction did you correctly flag? (This is just recall by another name.)<br>"
    "- <b>False Positive Rate (FPR)</b>: of all readings NOT from the target city, "
    "what fraction did you incorrectly flag?<br><br>"
    "Plot TPR (y-axis) vs FPR (x-axis) and you get the ROC curve. A perfect model "
    "shoots straight up to TPR=1.0 at FPR=0 -- it finds all the target city readings "
    "without any false alarms. A coin-flip model traces the diagonal. <b>AUC</b> "
    "(Area Under the Curve) compresses this into a single number: 0.5 = coin flip, "
    "1.0 = perfect separation."
)

formula_box(
    "TPR and FPR",
    r"\underbrace{\text{TPR}}_{\text{hit rate}} = \frac{\underbrace{TP}_{\text{correctly caught}}}{\underbrace{TP+FN}_{\text{all actual positive}}}, \qquad \underbrace{\text{FPR}}_{\text{false alarm rate}} = \frac{\underbrace{FP}_{\text{wrong alerts}}}{\underbrace{FP+TN}_{\text{all actual negative}}}",
    f"For our weather task: TPR answers 'of all true {target_city} readings, what "
    f"fraction did the model catch?' FPR answers 'of all readings from OTHER cities, "
    f"what fraction did the model incorrectly call {target_city}?'"
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

st.markdown(
    f"An AUC of {roc_auc:.3f} for {target_city} means: if you picked one random "
    f"weather reading from {target_city} and one random reading from another city, "
    f"there is a {roc_auc*100:.1f}% chance the model assigns a higher probability to "
    f"the actual {target_city} reading. That is the intuitive meaning of AUC -- it "
    f"measures how well the model ranks the target city's readings above everything else."
)

# Multi-class ROC (one-vs-rest)
st.subheader("Multi-Class ROC (One-vs-Rest)")

st.markdown(
    "Now let us see the ROC curve for every city at once. Each curve shows how well "
    "the random forest model can distinguish that city's weather from all the others. "
    "Cities with unique climates should have curves that hug the top-left corner "
    "(high AUC). Cities with weather siblings should have curves closer to the diagonal."
)

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
    "Look at which cities have the highest AUC. NYC typically stands out because its "
    "cold winters are unlike anything in Texas or coastal California -- when the model "
    "sees temperature below 0 degrees C with high pressure, it knows that is not Houston. "
    "LA stands out because of its dry, mild conditions -- low humidity readings that no "
    "Texas city produces. Dallas and San Antonio tend to have the lowest AUC because "
    "their weather overlaps so heavily. The model is basically reading a weather report "
    "and trying to guess which Texas city it came from, which is genuinely hard even for "
    "a meteorologist looking at the same four numbers."
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
    "If AUC = 0.5 for 'Is this reading from Los Angeles?', the model is performing:",
    [
        "Perfectly -- it identifies every LA reading",
        "No better than random guessing at distinguishing LA from other cities",
        "With high precision but low recall for LA",
        "With high recall but low precision for LA",
    ],
    correct_idx=1,
    explanation=(
        "AUC of 0.5 means the ROC curve is the diagonal line -- the model cannot "
        "distinguish LA weather from non-LA weather at any threshold. If you randomly "
        "picked an LA reading and a non-LA reading, the model would assign the higher "
        "probability to LA only 50% of the time -- exactly coin-flip odds. In practice, "
        "LA usually has a much higher AUC because its dry, mild climate is quite "
        "distinctive compared to the humid Texas cities and cold NYC winters."
    ),
    key="ch45_quiz1",
)

quiz(
    "Lowering the classification threshold for 'Is this reading from Dallas?' will generally:",
    [
        "Increase precision, decrease recall",
        "Increase recall, decrease precision",
        "Increase both precision and recall",
        "Have no effect on predictions",
    ],
    correct_idx=1,
    explanation=(
        "A lower threshold means the model says 'yes, Dallas' more often -- even when "
        "it is only 20% or 30% confident. This catches more actual Dallas readings that "
        "would have been missed at a higher threshold (higher recall), but it also sweeps "
        "in readings from San Antonio and Austin that happen to look a bit Dallas-like "
        "(lower precision). You are trading false negatives for false positives. Whether "
        "that is a good trade depends on the cost of each type of error in your specific "
        "application."
    ),
    key="ch45_quiz2",
)

quiz(
    "The model has precision = 0.95 and recall = 0.40 for NYC. What does this mean?",
    [
        "The model is bad at identifying NYC weather",
        "When it says NYC, it is almost always right, but it misses most NYC readings",
        "It identifies most NYC readings but makes many false NYC predictions",
        "The model has high accuracy for NYC",
    ],
    correct_idx=1,
    explanation=(
        "Precision of 0.95 means: of the times the model says 'this is NYC weather,' "
        "95% of the time it is correct. The model is very careful and only says NYC when "
        "it is highly confident -- maybe when it sees temperatures below freezing. But "
        "recall of 0.40 means it only catches 40% of actual NYC readings. The other 60% -- "
        "maybe NYC's milder spring and fall days, which overlap with other cities' temperatures -- "
        "get incorrectly assigned elsewhere. This model is like a cautious doctor who only "
        "diagnoses a condition when absolutely certain, but consequently misses many real cases."
    ),
    key="ch45_quiz3",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "The **confusion matrix** shows exactly which cities the model confuses with each other. Dallas-San Antonio confusion reveals meteorological similarity; NYC-LA separation reveals climatic distinctiveness.",
    "**Precision** answers 'when the model says this reading is from Houston, how often is it right?' **Recall** answers 'of all the readings that truly came from Houston, how many did the model find?' These are fundamentally different questions.",
    "**F1-Score** balances precision and recall into a single number -- useful when you cannot decide which error type is worse, but it hides the tradeoff.",
    "**Threshold tuning** lets you slide between 'catch everything, tolerate false alarms' and 'only flag what you are sure about, miss some real cases.' The right threshold depends on the cost of each error in your application.",
    "The **ROC curve** and **AUC** measure overall separability across all thresholds. AUC = 0.5 means the model cannot tell the target city from the rest; AUC = 1.0 means perfect separation.",
    "Cities with unique climates (NYC's cold, LA's dryness) get high AUC. Weather siblings like Dallas and San Antonio get lower AUC -- the model is telling you their climates genuinely overlap.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 44: Bias-Variance Tradeoff",
    prev_page="44_Bias_Variance_Tradeoff.py",
    next_label="Ch 46: Regression Metrics",
    next_page="46_Regression_Metrics.py",
)
