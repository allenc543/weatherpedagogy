"""Chapter 21: Logistic Regression -- From sigmoid to multi-class city prediction."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, scatter_chart
from utils.ml_helpers import prepare_classification_data, classification_metrics, plot_confusion_matrix
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(21, "Logistic Regression", part="V")
st.markdown(
    "Here is a fun naming disaster in machine learning: logistic **regression** is "
    "not a regression algorithm. It is a classification algorithm wearing a regression "
    "algorithm's nametag at a conference. The name stuck because the math under the "
    "hood involves regressing onto log-odds, but what it actually *does* is draw a "
    "line between categories and say 'you belong on this side.' We will start with "
    "the simplest version -- NYC vs LA, which is basically asking 'is it cold and "
    "humid, or warm and dry?' -- and then work our way up to the full 6-city problem, "
    "which is considerably harder and more interesting."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
filt = sidebar_filters(df)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 -- The Sigmoid Function
# ══════════════════════════════════════════════════════════════════════════════
st.header("1. The Sigmoid Function")

concept_box(
    "From Linear to Probability",
    "Linear regression can spit out any number between negative infinity and positive "
    "infinity, which is deeply unhelpful if you are trying to answer a yes-or-no question. "
    "'Is this New York weather?' 'The model says 7.3.' What does that even mean?<br><br>"
    "Logistic regression solves this by taking that unbounded number and squishing it "
    "through the <b>sigmoid function</b>, which maps everything to the range (0, 1). "
    "Now '0.93' means 'I am 93% confident this is NYC weather,' and we can work with that."
)

formula_box(
    "Sigmoid Function",
    r"\underbrace{\sigma(z)}_{\text{predicted probability}} = \frac{1}{1 + e^{-\underbrace{z}_{\text{linear combo}}}} \quad\text{where } z = \underbrace{\mathbf{w}^\top}_{\text{learned weights}} \underbrace{\mathbf{x}}_{\text{weather features}} + \underbrace{b}_{\text{bias term}}",
    "The output is always between 0 and 1, which we interpret as P(y=1|x). "
    "Negative infinity maps to 0, positive infinity maps to 1, and 0 maps to exactly 0.5. "
    "It is one of those elegant mathematical objects that does exactly what you need."
)

# Interactive sigmoid plot
st.subheader("Interactive Sigmoid Curve")
shift = st.slider("Shift the sigmoid (bias b)", -5.0, 5.0, 0.0, 0.5, key="sigmoid_shift")
stretch = st.slider("Stretch the sigmoid (weight w)", 0.2, 5.0, 1.0, 0.2, key="sigmoid_stretch")

z_vals = np.linspace(-10, 10, 300)
sig_vals = 1.0 / (1.0 + np.exp(-(stretch * z_vals + shift)))

fig_sig = go.Figure()
fig_sig.add_trace(go.Scatter(x=z_vals, y=sig_vals, mode="lines", name="sigmoid",
                              line=dict(color="#2E86C1", width=3)))
fig_sig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Decision Boundary (0.5)")
apply_common_layout(fig_sig, title="Sigmoid Function", height=400)
fig_sig.update_layout(xaxis_title="z = wTx + b", yaxis_title="P(y=1|x)")
st.plotly_chart(fig_sig, use_container_width=True)

insight_box(
    "Play with the sliders and notice something: when z = 0, the sigmoid outputs exactly "
    "0.5 -- the point of maximum uncertainty. Crank the weight w up toward 5 and "
    "the sigmoid becomes a near-vertical cliff: the model is saying 'I am VERY sure "
    "about which side you are on.' Pull it down to 0.2 and the transition is gentle, "
    "almost apologetic. This is the model's confidence knob."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 -- Binary Classification: NYC vs LA
# ══════════════════════════════════════════════════════════════════════════════
st.header("2. Binary Classification: NYC vs Los Angeles")

st.markdown(
    "Let's start with the easy version. NYC and LA have about as different a climate "
    "as two major US cities can have: one gets blizzards, the other gets 75-degree "
    "Christmases. If our algorithm cannot tell these apart, we should give up and "
    "go home. Fortunately, it can."
)

df_binary = df[df["city"].isin(["NYC", "Los Angeles"])].copy()

features_selected = st.multiselect(
    "Select features for binary model",
    FEATURE_COLS,
    default=FEATURE_COLS,
    key="bin_features"
)

if len(features_selected) < 1:
    st.warning("Please select at least one feature.")
    st.stop()

X_train_b, X_test_b, y_train_b, y_test_b, le_b, scaler_b = prepare_classification_data(
    df_binary, features_selected, target="city", test_size=0.2, scale=True, seed=42
)

C_bin = st.slider("Regularization strength C (binary)", 0.01, 10.0, 1.0, 0.1, key="c_bin")

model_bin = LogisticRegression(C=C_bin, max_iter=1000, random_state=42)
model_bin.fit(X_train_b, y_train_b)
y_pred_b = model_bin.predict(X_test_b)
labels_b = le_b.classes_.tolist()
metrics_b = classification_metrics(y_test_b, y_pred_b, labels=labels_b)

col1, col2 = st.columns(2)
col1.metric("Binary Accuracy", f"{metrics_b['accuracy']:.1%}")
col2.metric("Test Samples", len(y_test_b))

st.plotly_chart(
    plot_confusion_matrix(metrics_b["confusion_matrix"], labels_b),
    use_container_width=True
)

insight_box(
    f"NYC vs LA achieves **{metrics_b['accuracy']:.1%}** accuracy. I want to be clear about "
    "what this means: a simple linear boundary in weather-feature space is enough to "
    "correctly identify which coast a weather observation comes from the vast majority "
    "of the time. These cities are just that different. Enjoy this feeling of success, "
    "because the Texas cities are about to make things much harder."
)

# Decision boundary visualization (2D)
st.subheader("2D Decision Boundary")
feat_x = st.selectbox("X-axis feature", FEATURE_COLS, index=0, key="db_x")
feat_y = st.selectbox("Y-axis feature", FEATURE_COLS, index=3, key="db_y")

model_2d = LogisticRegression(C=C_bin, max_iter=1000, random_state=42)
X_2d = df_binary[[feat_x, feat_y]].dropna()
y_2d = le_b.transform(df_binary.loc[X_2d.index, "city"])
model_2d.fit(X_2d, y_2d)

# Create mesh grid
x_min, x_max = X_2d[feat_x].min() - 1, X_2d[feat_x].max() + 1
y_min, y_max = X_2d[feat_y].min() - 1, X_2d[feat_y].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = model_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

fig_db = go.Figure()
fig_db.add_trace(go.Contour(
    x=np.linspace(x_min, x_max, 200), y=np.linspace(y_min, y_max, 200),
    z=Z, colorscale="RdBu", opacity=0.5, showscale=True,
    contours=dict(showlines=False),
    colorbar=dict(title="P(class 1)")
))
sample = df_binary.sample(min(2000, len(df_binary)), random_state=42)
for city in ["NYC", "Los Angeles"]:
    city_data = sample[sample["city"] == city]
    fig_db.add_trace(go.Scatter(
        x=city_data[feat_x], y=city_data[feat_y],
        mode="markers", name=city, opacity=0.4,
        marker=dict(color=CITY_COLORS[city], size=4)
    ))
apply_common_layout(fig_db, title="Logistic Regression Decision Boundary", height=500)
fig_db.update_layout(
    xaxis_title=FEATURE_LABELS.get(feat_x, feat_x),
    yaxis_title=FEATURE_LABELS.get(feat_y, feat_y)
)
st.plotly_chart(fig_db, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 -- Multi-class: All 6 Cities (One-vs-Rest)
# ══════════════════════════════════════════════════════════════════════════════
st.header("3. Multi-class Classification: All 6 Cities")

concept_box(
    "One-vs-Rest (OvR) Strategy",
    "Logistic regression was born to answer binary questions: yes or no, this or that. "
    "So how do you get it to choose between 6 cities? You might object: why not just "
    "train one big model? Fair point, but the math is cleaner with a trick called "
    "<b>One-vs-Rest</b>: train 6 separate binary classifiers, each asking 'is this "
    "NYC, or literally anything else?' Then pick the one that is most confident."
)

formula_box(
    "Multi-class Decision Rule",
    r"\underbrace{\hat{y}}_{\text{predicted city}} = \underbrace{\arg\max_k}_{\text{pick best class}} \; \underbrace{P(y=k \mid \mathbf{x})}_{\text{class probability}} = \arg\max_k \; \underbrace{\sigma(\mathbf{w}_k^\top \mathbf{x} + b_k)}_{\text{sigmoid of city k's score}}",
    "Each class gets its own weight vector and bias. It is like 6 experts, each "
    "specialized in recognizing one city, shouting their confidence levels. Loudest wins."
)

C_multi = st.slider("Regularization strength C (multi-class)", 0.01, 10.0, 1.0, 0.1, key="c_multi")

X_train_m, X_test_m, y_train_m, y_test_m, le_m, scaler_m = prepare_classification_data(
    filt, FEATURE_COLS, target="city", test_size=0.2, scale=True, seed=42
)

model_multi = LogisticRegression(
    C=C_multi, max_iter=1000, multi_class="ovr", random_state=42
)
model_multi.fit(X_train_m, y_train_m)
y_pred_m = model_multi.predict(X_test_m)
labels_m = le_m.classes_.tolist()
metrics_m = classification_metrics(y_test_m, y_pred_m, labels=labels_m)

col1, col2, col3 = st.columns(3)
col1.metric("6-City Accuracy", f"{metrics_m['accuracy']:.1%}")
col2.metric("Train Samples", len(y_train_m))
col3.metric("Test Samples", len(y_test_m))

st.plotly_chart(
    plot_confusion_matrix(metrics_m["confusion_matrix"], labels_m),
    use_container_width=True
)

st.markdown("**Per-class precision and recall:**")
report_df = pd.DataFrame(metrics_m["report"]).T
report_df = report_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
report_df = report_df[["precision", "recall", "f1-score", "support"]]
report_df = report_df.round(3)
st.dataframe(report_df, use_container_width=True)

warning_box(
    "Look at the confusion matrix and you will see a familiar problem: the Texas cities "
    "(Dallas, Houston, San Antonio, Austin) keep getting confused with each other. "
    "This is not a failure of the algorithm -- it is a failure of geography. These cities "
    "are all in the same state, with the same subtropical climate, and their weather "
    "genuinely looks similar. NYC and LA, meanwhile, are basically on different planets "
    "weather-wise."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 -- Feature Importance from Coefficients
# ══════════════════════════════════════════════════════════════════════════════
st.header("4. Feature Importance from Coefficients")

concept_box(
    "Interpreting Logistic Regression Coefficients",
    "Here is what is delightful about logistic regression compared to the black-box "
    "models we will see later: you can just <i>read</i> what it learned. Each "
    "coefficient tells you 'for every one-unit increase in this feature (after "
    "scaling), the log-odds of this class change by this much.' Large positive "
    "coefficient? The model thinks that feature is strong evidence for this city. "
    "Large negative? Strong evidence against. Zero? The model does not care about it."
)

coef_df = pd.DataFrame(
    model_multi.coef_,
    columns=[FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
    index=labels_m
)

fig_coef = px.imshow(
    coef_df.values,
    x=coef_df.columns.tolist(),
    y=coef_df.index.tolist(),
    color_continuous_scale="RdBu_r",
    aspect="auto",
    title="Logistic Regression Coefficients by City and Feature",
    labels=dict(color="Coefficient")
)
apply_common_layout(fig_coef, title="Logistic Regression Coefficients by City and Feature", height=400)
st.plotly_chart(fig_coef, use_container_width=True)

# Absolute feature importance (mean abs coeff across classes)
mean_abs_coef = np.abs(model_multi.coef_).mean(axis=0)
feat_imp_df = pd.DataFrame({
    "Feature": [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
    "Mean |Coefficient|": mean_abs_coef
}).sort_values("Mean |Coefficient|", ascending=True)

fig_imp = px.bar(
    feat_imp_df, x="Mean |Coefficient|", y="Feature",
    orientation="h", title="Overall Feature Importance (Mean Absolute Coefficient)",
    color="Mean |Coefficient|", color_continuous_scale="Blues"
)
apply_common_layout(fig_imp, title="Overall Feature Importance (Mean Absolute Coefficient)", height=350)
st.plotly_chart(fig_imp, use_container_width=True)

insight_box(
    "Surface pressure and temperature dominate. This makes physical sense: pressure "
    "correlates with elevation and large-scale weather systems, while temperature "
    "captures latitude and seasonal differences. The model has essentially rediscovered "
    "geography from weather data alone, which is a nice sanity check."
)

code_example("""
from sklearn.linear_model import LogisticRegression

# Binary: NYC vs LA
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Multi-class: One-vs-Rest
model_ovr = LogisticRegression(C=1.0, multi_class='ovr', max_iter=1000)
model_ovr.fit(X_train, y_train)

# Coefficients = feature importance
print(model_ovr.coef_)         # shape: (n_classes, n_features)
print(model_ovr.intercept_)    # shape: (n_classes,)
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 -- Quiz & Takeaways
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

quiz(
    "What does the sigmoid function guarantee about its output?",
    [
        "The output is always between -1 and 1",
        "The output is always between 0 and 1",
        "The output is always an integer",
        "The output is always positive",
    ],
    correct_idx=1,
    explanation="The sigmoid function squishes any real number into the (0, 1) interval. "
    "This is exactly what makes it useful for modeling probabilities -- you cannot have "
    "a 150% chance of rain, and the sigmoid makes sure that never happens.",
    key="q_logistic_1"
)

quiz(
    "In One-vs-Rest (OvR) logistic regression with 6 classes, how many binary classifiers are trained?",
    ["1", "3", "6", "15"],
    correct_idx=2,
    explanation="One per class. Each classifier learns to distinguish its city from all the "
    "others. With 6 cities, that is 6 binary classifiers, each a specialist. (If you "
    "guessed 15, you were thinking of One-vs-One, which is a different approach.)",
    key="q_logistic_2"
)

takeaways([
    "Logistic regression maps a linear combination through the sigmoid to produce probabilities -- it is the simplest way to get from 'weighted sum of features' to 'probability of a class.'",
    "The decision boundary lives where P(y=1|x) = 0.5 -- the point of maximum uncertainty.",
    "Multi-class problems use One-vs-Rest: one specialized binary classifier per class, loudest confidence wins.",
    "Coefficients are directly interpretable as feature importance (after scaling), which is a luxury most ML models do not give you.",
    "NYC vs LA is an easy binary problem; the 4 Texas cities are much harder because geography made their weather similar.",
    "The regularization parameter C is your overfitting dial: high C = trust the training data more, low C = keep weights small and play it safe.",
])
