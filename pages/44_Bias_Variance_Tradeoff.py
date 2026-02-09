"""Chapter 44: Bias-Variance Tradeoff -- Underfitting, overfitting, learning curves."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(44, "Bias-Variance Tradeoff", part="X")
st.markdown(
    "The beautiful thing about the bias-variance tradeoff is that it explains "
    "basically every mistake a predictive model can make, and it does so with a "
    "clean mathematical decomposition. Every model walks a tightrope: lean too far "
    "toward simplicity and you **underfit** (high bias), lean too far toward "
    "complexity and you **overfit** (high variance). Understanding this tradeoff "
    "is the single most important conceptual insight in all of machine learning."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 44.1 Concepts ────────────────────────────────────────────────────────────
st.header("44.1  Bias, Variance, and Noise")

concept_box(
    "The Decomposition",
    "For any model, the expected prediction error breaks down into exactly three pieces: "
    "<br><b>Error = Bias-squared + Variance + Irreducible Noise</b><br>"
    "<b>Bias</b>: the error that comes from wrong assumptions. If you fit a straight line "
    "to a sine wave, you have high bias -- your model is structurally incapable of "
    "capturing the truth. "
    "<b>Variance</b>: the error that comes from being too sensitive to the particular "
    "training data you happened to use. A model that memorizes every training point has "
    "high variance -- train it on a slightly different sample and you get a completely "
    "different model. "
    "<b>Noise</b>: the inherent randomness in the data that no model, no matter how "
    "perfect, could ever capture. This is the universe's way of keeping us humble."
)

formula_box(
    "Bias-Variance Decomposition",
    r"\mathbb{E}\!\left[(y - \hat{f}(x))^2\right] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2",
    "Sigma-squared is the irreducible error -- the noise baked into the data-generating process. You cannot beat it. You can only waste your time trying."
)

# Visual illustration
st.subheader("Underfitting vs Overfitting")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Underfitting (High Bias)**")
    st.markdown(
        "- Model too simple to capture reality\n"
        "- Poor on training AND test data\n"
        "- Like describing all of human emotion as 'good' or 'bad'\n"
        "- Example: a linear model for seasonal temperature curves"
    )
with col2:
    st.markdown("**Just Right**")
    st.markdown(
        "- Captures the real pattern without memorizing noise\n"
        "- Good on both training and test data\n"
        "- The Goldilocks zone of complexity\n"
        "- Example: a moderate-depth decision tree"
    )
with col3:
    st.markdown("**Overfitting (High Variance)**")
    st.markdown(
        "- Model so complex it memorizes the noise\n"
        "- Perfect on training, terrible on test\n"
        "- Like a student who memorized every typo in the textbook\n"
        "- Example: a polynomial that wiggles through every data point"
    )

# ── 44.2 Polynomial Degree Slider ───────────────────────────────────────────
st.header("44.2  Interactive: Polynomial Fit on Annual Temperature Curve")

concept_box(
    "Polynomial Regression as Complexity Knob",
    "Polynomial degree is the perfect toy example for the bias-variance tradeoff "
    "because it gives us a single knob to turn. Degree 1 is a straight line -- almost "
    "certainly too simple to capture a seasonal temperature curve (high bias). "
    "Degree 20 is a wiggly monstrosity that threads through every data point like "
    "a terrified snake (high variance). Somewhere in between lies the truth."
)

poly_city = st.selectbox("City", CITY_LIST, key="poly_city")
city_annual = fdf[fdf["city"] == poly_city].groupby("day_of_year")["temperature_c"].mean().reset_index()
city_annual.columns = ["day_of_year", "avg_temp"]

poly_degree = st.slider("Polynomial Degree (complexity)", 1, 20, 3, key="poly_deg")

X_day = city_annual["day_of_year"].values.reshape(-1, 1)
y_temp = city_annual["avg_temp"].values

# Train-test split for the annual curve
X_tr, X_te, y_tr, y_te = train_test_split(X_day, y_temp, test_size=0.3, random_state=42)

poly_model = make_pipeline(PolynomialFeatures(poly_degree), LinearRegression())
poly_model.fit(X_tr, y_tr)

train_rmse = np.sqrt(mean_squared_error(y_tr, poly_model.predict(X_tr)))
test_rmse = np.sqrt(mean_squared_error(y_te, poly_model.predict(X_te)))

# Generate smooth prediction line
X_smooth = np.linspace(1, 365, 365).reshape(-1, 1)
y_smooth = poly_model.predict(X_smooth)

fig_poly = go.Figure()
fig_poly.add_trace(go.Scatter(
    x=city_annual["day_of_year"], y=city_annual["avg_temp"],
    mode="markers", marker=dict(size=3, color="#2A9D8F", opacity=0.5),
    name="Daily Mean Temp",
))
fig_poly.add_trace(go.Scatter(
    x=X_smooth.flatten(), y=y_smooth,
    mode="lines", line=dict(color="#E63946", width=2),
    name=f"Degree {poly_degree} Polynomial",
))
apply_common_layout(fig_poly, title=f"Polynomial Degree {poly_degree} Fit -- {poly_city}", height=450)
fig_poly.update_layout(xaxis_title="Day of Year", yaxis_title="Temperature (C)")
st.plotly_chart(fig_poly, use_container_width=True)

c1, c2 = st.columns(2)
c1.metric("Train RMSE", f"{train_rmse:.3f} C")
c2.metric("Test RMSE", f"{test_rmse:.3f} C", delta=f"{test_rmse - train_rmse:.3f} C")

if poly_degree <= 2:
    st.info("This low-degree polynomial is the model equivalent of describing seasons as 'sometimes warm, sometimes cold.' It **underfits** -- it cannot capture the seasonal curve.")
elif poly_degree <= 6:
    st.success("This polynomial degree hits the sweet spot -- enough complexity to capture the seasonal pattern without going off the rails on the test data.")
else:
    st.warning("Watch that test RMSE diverge from train RMSE -- the polynomial is starting to **overfit**, contorting itself to match noise in the training data rather than learning the underlying seasonal pattern.")

# ── 44.3 Training Error vs Test Error vs Complexity ─────────────────────────
st.header("44.3  Training Error vs Test Error as Complexity Increases")

max_degree_range = st.slider("Max polynomial degree to evaluate", 5, 25, 15, key="max_deg")
degrees = list(range(1, max_degree_range + 1))
train_errors, test_errors = [], []

for d in degrees:
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(X_tr, y_tr)
    train_errors.append(np.sqrt(mean_squared_error(y_tr, model.predict(X_tr))))
    test_errors.append(np.sqrt(mean_squared_error(y_te, model.predict(X_te))))

fig_bv = go.Figure()
fig_bv.add_trace(go.Scatter(x=degrees, y=train_errors, mode="lines+markers",
                             name="Train RMSE", line=dict(color="#2A9D8F")))
fig_bv.add_trace(go.Scatter(x=degrees, y=test_errors, mode="lines+markers",
                             name="Test RMSE", line=dict(color="#E63946")))
# Clip y-axis to avoid showing extremely large overfitting values
y_max = min(max(test_errors), np.mean(test_errors) * 3)
apply_common_layout(fig_bv, title="Bias-Variance Tradeoff: Error vs Complexity", height=450)
fig_bv.update_layout(
    xaxis_title="Polynomial Degree (Model Complexity)",
    yaxis_title="RMSE (C)",
    yaxis_range=[0, y_max],
)
st.plotly_chart(fig_bv, use_container_width=True)

insight_box(
    "This is the most important chart in all of machine learning. The training error "
    "always decreases with complexity -- of course it does, you are giving the model "
    "more knobs to turn. But the test error follows a U-shape: first decreasing "
    "(reducing bias), then increasing (growing variance). The **sweet spot** -- the "
    "bottom of that U -- is where your model understands the signal without memorizing "
    "the noise."
)

# Find optimal degree
best_idx = np.argmin(test_errors)
st.success(f"Optimal polynomial degree: **{degrees[best_idx]}** with test RMSE = {test_errors[best_idx]:.3f} C")

# ── 44.4 Learning Curves ────────────────────────────────────────────────────
st.header("44.4  Learning Curves: Accuracy vs Training Set Size")

concept_box(
    "Learning Curves",
    "A <b>learning curve</b> answers one of the most practical questions in ML: "
    "'Should I go collect more data?' Plot model performance against training set "
    "size and you get your answer: <br>"
    "- <b>High bias</b>: both training and validation scores plateau at a low value. "
    "More data will not help -- your model is too simple. Go get a better model. <br>"
    "- <b>High variance</b>: big gap between training and validation. More data "
    "will likely help -- your model is capable but data-hungry. Go get more data."
)

st.subheader("City Classification Learning Curve")
lc_depth = st.slider("Random Forest max_depth (complexity)", 1, 30, 10, key="lc_depth")

le = LabelEncoder()
X_lc = fdf[FEATURE_COLS].dropna()
y_lc = le.fit_transform(fdf.loc[X_lc.index, "city"])

# Subsample for speed
sample_n = min(6000, len(X_lc))
rng = np.random.RandomState(42)
idx = rng.choice(len(X_lc), sample_n, replace=False)
X_lc_s = X_lc.iloc[idx].values
y_lc_s = y_lc[idx]

rf_lc = RandomForestClassifier(n_estimators=50, max_depth=lc_depth, random_state=42, n_jobs=-1)

with st.spinner("Computing learning curves..."):
    train_sizes_abs, train_scores, val_scores = learning_curve(
        rf_lc, X_lc_s, y_lc_s,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring="accuracy", n_jobs=-1,
    )

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

fig_lc = go.Figure()
fig_lc.add_trace(go.Scatter(
    x=train_sizes_abs, y=train_mean,
    mode="lines+markers", name="Training Score",
    line=dict(color="#2A9D8F"),
    error_y=dict(type="data", array=train_std, visible=True),
))
fig_lc.add_trace(go.Scatter(
    x=train_sizes_abs, y=val_mean,
    mode="lines+markers", name="Validation Score",
    line=dict(color="#E63946"),
    error_y=dict(type="data", array=val_std, visible=True),
))
apply_common_layout(fig_lc, title=f"Learning Curve (max_depth={lc_depth})", height=450)
fig_lc.update_layout(xaxis_title="Training Set Size", yaxis_title="Accuracy")
st.plotly_chart(fig_lc, use_container_width=True)

gap = train_mean[-1] - val_mean[-1]
if gap > 0.1:
    st.warning(
        f"The gap between training ({train_mean[-1]:.3f}) and validation ({val_mean[-1]:.3f}) "
        f"accuracy is **{gap:.3f}** -- classic high variance territory. Your model is "
        "memorizing the training data and struggling to generalize. Try reducing max_depth, "
        "adding regularization, or (the boring but effective answer) collecting more data."
    )
elif val_mean[-1] < 0.5:
    st.info(
        f"Validation accuracy is only {val_mean[-1]:.3f} -- your model is underfitting. "
        "It is too simple to capture the real patterns. Try increasing max_depth, "
        "using more features, or switching to a more expressive model."
    )
else:
    st.success(
        f"Training: {train_mean[-1]:.3f}, Validation: {val_mean[-1]:.3f}. "
        "The model has found a reasonable bias-variance balance. Not too hot, not too cold."
    )

# Compare different complexities
st.subheader("Learning Curves at Different Complexities")
depths_to_compare = [1, 5, 15, None]
fig_multi = go.Figure()
colors = ["#7209B7", "#FB8500", "#2A9D8F", "#E63946"]

for depth, color in zip(depths_to_compare, colors):
    rf_temp = RandomForestClassifier(n_estimators=50, max_depth=depth, random_state=42, n_jobs=-1)
    _, t_sc, v_sc = learning_curve(
        rf_temp, X_lc_s, y_lc_s,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=3, scoring="accuracy", n_jobs=-1,
    )
    label = f"depth={depth}" if depth else "depth=unlimited"
    fig_multi.add_trace(go.Scatter(
        x=train_sizes_abs[:8] if len(train_sizes_abs) >= 8 else train_sizes_abs,
        y=v_sc.mean(axis=1),
        mode="lines+markers", name=f"Val {label}",
        line=dict(color=color),
    ))

apply_common_layout(fig_multi, title="Validation Curves at Different Depths", height=450)
fig_multi.update_layout(xaxis_title="Training Set Size", yaxis_title="Validation Accuracy")
st.plotly_chart(fig_multi, use_container_width=True)

insight_box(
    "Shallow trees (depth=1) plateau quickly at low accuracy -- they have hit the "
    "ceiling of their simplicity and no amount of data can save them. Very deep trees "
    "may overfit on small datasets but improve with more data as the variance gets "
    "averaged away. The right depth depends on how much data you have. This is why "
    "'it depends' is the only honest answer to most ML questions."
)

code_example("""from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy'
)

# Plot mean +/- std
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "A model with high bias and low variance is likely:",
    [
        "Overfitting the training data",
        "Underfitting the training data",
        "Perfectly balanced",
        "Memorising noise",
    ],
    correct_idx=1,
    explanation="High bias means the model is making strong (wrong) assumptions that are too simple to capture reality. It is underfitting -- performing poorly on both training and test data.",
    key="ch44_quiz1",
)

quiz(
    "If training accuracy is 99% but test accuracy is 60%, the model suffers from:",
    [
        "High bias",
        "High variance (overfitting)",
        "Irreducible error",
        "Data leakage",
    ],
    correct_idx=1,
    explanation="A 39-point gap between training and test performance is the smoking gun of overfitting. The model has memorized the training data (it gets 99% there) but learned almost nothing generalizable (60% on new data).",
    key="ch44_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "**Bias** = your model is too simple and misses the pattern. **Variance** = your model is too sensitive and memorizes the noise. Both are bad in different ways.",
    "Total error = Bias-squared + Variance + Irreducible Noise. You cannot eliminate noise, so your job is to minimize the sum of bias and variance.",
    "Training error always decreases with complexity (by construction). Test error follows a U-shape -- and the bottom of the U is where you want to live.",
    "**Learning curves** are your diagnostic tool: they tell you whether you need more data (high variance) or a more complex model (high bias).",
    "The optimal model minimizes **test error**. It sits at the sweet spot between 'too simple to learn anything' and 'so complex it memorizes everything.'",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 43: Cross-Validation",
    prev_page="43_Cross_Validation.py",
    next_label="Ch 45: ROC/AUC & Confusion Matrix",
    next_page="45_ROC_AUC_Confusion_Matrix.py",
)
