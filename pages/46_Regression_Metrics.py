"""Chapter 46: Regression Metrics -- MSE, RMSE, MAE, R-squared, residual analysis."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.ml_helpers import prepare_regression_data, regression_metrics
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(46, "Regression Metrics", part="X")
st.markdown(
    "When your model predicts a continuous number instead of a category, you need "
    "different tools to judge how wrong it is. The question is not 'did you get it "
    "right or wrong?' but 'how far off were you, and in what way?' This chapter "
    "covers **MSE, RMSE, MAE, and R-squared** -- each of which answers a slightly "
    "different version of that question -- and introduces **residual analysis**, "
    "which is what you do when you want to know not just *how much* the model is "
    "wrong, but *where* and *why*."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 46.1 Metric Definitions ─────────────────────────────────────────────────
st.header("46.1  Regression Metric Definitions")

concept_box(
    "Regression Metrics Overview",
    "All regression metrics are different ways of answering 'how far off are the "
    "predictions from reality?' But the devil is in the details:<br>"
    "- <b>MAE</b>: the average absolute error. Intuitive, robust to outliers, in "
    "the original units. The 'average miss distance.'<br>"
    "- <b>MSE</b>: the average squared error. Punishes large errors disproportionately, "
    "which is sometimes what you want (a 10-degree error is arguably more than twice "
    "as bad as a 5-degree error).<br>"
    "- <b>RMSE</b>: the square root of MSE. Back in original units, so you can "
    "actually interpret it.<br>"
    "- <b>R-squared</b>: the fraction of variance explained. 1.0 = your model is "
    "perfect, 0.0 = you might as well have predicted the mean every time."
)

col1, col2 = st.columns(2)
with col1:
    formula_box(
        "Mean Absolute Error (MAE)",
        r"\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|",
        "The average magnitude of errors, ignoring direction. If your MAE is 3 C, your model is typically off by about 3 degrees."
    )
    formula_box(
        "Mean Squared Error (MSE)",
        r"\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2",
        "Squaring means a prediction that is off by 10 degrees contributes 100 to the MSE, while one off by 1 degree contributes only 1. This metric really hates big mistakes."
    )
with col2:
    formula_box(
        "Root Mean Squared Error (RMSE)",
        r"\text{RMSE} = \sqrt{\text{MSE}}",
        "Same units as the target variable. RMSE of 4.2 C means your typical error is about 4 degrees. This is the metric most people find easiest to reason about."
    )
    formula_box(
        "R-squared (Coefficient of Determination)",
        r"R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}",
        "1.0 = perfect prediction, 0.0 = you are doing no better than predicting the average every single time. Can actually go negative if your model is worse than the mean."
    )

# ── 46.2 Compare Models ─────────────────────────────────────────────────────
st.header("46.2  Compare Metrics Across Regression Models")

st.markdown(
    "Let us predict **temperature** from other weather features and see how different "
    "models stack up. This is a nice test because temperature has clear physical meaning -- "
    "an RMSE of 3 degrees Celsius is something you can actually feel."
)

target = "temperature_c"
features_reg = ["relative_humidity_pct", "wind_speed_kmh", "surface_pressure_hpa", "hour", "month"]

sample_size = min(10000, len(fdf))
sample_df = fdf.sample(sample_size, random_state=42) if len(fdf) > sample_size else fdf

X_train, X_test, y_train, y_test, _ = prepare_regression_data(
    sample_df, features_reg, target, test_size=0.2
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest (depth=5)": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1),
    "Random Forest (depth=15)": RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
}

results = []
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    m = regression_metrics(y_test, y_pred)
    m["Model"] = name
    results.append(m)
    trained_models[name] = (model, y_pred)

results_df = pd.DataFrame(results)[["Model", "mae", "rmse", "mse", "r2"]]
results_df.columns = ["Model", "MAE (C)", "RMSE (C)", "MSE (C^2)", "R-squared"]
results_df = results_df.sort_values("RMSE (C)")

st.dataframe(
    results_df.style.format({
        "MAE (C)": "{:.3f}", "RMSE (C)": "{:.3f}",
        "MSE (C^2)": "{:.3f}", "R-squared": "{:.4f}",
    }).highlight_min(subset=["MAE (C)", "RMSE (C)", "MSE (C^2)"], color="#d4edda")
     .highlight_max(subset=["R-squared"], color="#d4edda"),
    use_container_width=True, hide_index=True,
)

# Bar chart comparison
fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(
    x=results_df["Model"], y=results_df["RMSE (C)"],
    name="RMSE", marker_color="#E63946",
))
fig_comp.add_trace(go.Bar(
    x=results_df["Model"], y=results_df["MAE (C)"],
    name="MAE", marker_color="#2A9D8F",
))
apply_common_layout(fig_comp, title="RMSE and MAE Across Models", height=450)
fig_comp.update_layout(barmode="group", yaxis_title="Error (C)", xaxis_tickangle=-30)
st.plotly_chart(fig_comp, use_container_width=True)

# ── 46.3 Physical Interpretation ────────────────────────────────────────────
st.header("46.3  Physical Interpretation of Metrics")

best_model_name = results_df.iloc[0]["Model"]
best_rmse = results_df.iloc[0]["RMSE (C)"]
best_mae = results_df.iloc[0]["MAE (C)"]

st.markdown(f"""
The best model (**{best_model_name}**) achieves:
- **RMSE = {best_rmse:.2f} C** -- on average, predictions are off by about {best_rmse:.1f} degrees Celsius. That is the difference between needing a light jacket and not.
- **MAE = {best_mae:.2f} C** -- the typical prediction misses by about {best_mae:.1f} degrees.

But what does 'off by {best_rmse:.1f} degrees' actually mean in context? Let us look at the temperature ranges we are dealing with:
""")

# Temperature range by city
temp_range = fdf.groupby("city")["temperature_c"].agg(["min", "max", "std"]).round(1)
temp_range.columns = ["Min Temp (C)", "Max Temp (C)", "Std Dev (C)"]
temp_range["Range (C)"] = temp_range["Max Temp (C)"] - temp_range["Min Temp (C)"]
st.dataframe(temp_range, use_container_width=True)

insight_box(
    f"An RMSE of {best_rmse:.1f} C means the model's typical error is about "
    f"**{(best_rmse / temp_range['Range (C)'].mean() * 100):.1f}%** of the average "
    f"city temperature range ({temp_range['Range (C)'].mean():.0f} C). "
    "That is good enough that you would trust it for deciding what to wear, "
    "but not for calibrating a scientific instrument. Context matters -- "
    "always interpret your metrics in the units of your problem."
)

# ── 46.4 Residual Plots ─────────────────────────────────────────────────────
st.header("46.4  Residual Analysis")

concept_box(
    "What Are Residuals?",
    "Residuals = Actual - Predicted. They are the model's mistakes, laid bare. "
    "Good residuals should be:<br>"
    "- <b>Randomly scattered</b> around zero (no pattern -- if you see a pattern, "
    "the model is systematically missing something it could learn).<br>"
    "- <b>Normally distributed</b> (bell-shaped histogram -- the errors should be "
    "symmetric, not skewed in one direction).<br>"
    "- <b>Constant variance</b> (homoscedasticity -- the model should not be more "
    "wrong at certain prediction ranges than others)."
)

model_choice = st.selectbox("Model for residual analysis", list(trained_models.keys()), key="resid_model")
_, y_pred_chosen = trained_models[model_choice]
residuals = y_test.values - y_pred_chosen

col_a, col_b = st.columns(2)
with col_a:
    # Residuals vs Predicted
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(
        x=y_pred_chosen, y=residuals,
        mode="markers", marker=dict(size=3, color="#2A9D8F", opacity=0.3),
        name="Residuals",
    ))
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
    apply_common_layout(fig_resid, title="Residuals vs Predicted", height=400)
    fig_resid.update_layout(xaxis_title="Predicted Temperature (C)", yaxis_title="Residual (C)")
    st.plotly_chart(fig_resid, use_container_width=True)

with col_b:
    # Residual histogram
    fig_hist = px.histogram(
        x=residuals, nbins=60,
        title="Residual Distribution",
        labels={"x": "Residual (C)", "count": "Count"},
        color_discrete_sequence=["#264653"],
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
    apply_common_layout(fig_hist, height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

# Actual vs Predicted
fig_ap = go.Figure()
fig_ap.add_trace(go.Scatter(
    x=y_test.values, y=y_pred_chosen,
    mode="markers", marker=dict(size=3, color="#E63946", opacity=0.2),
    name="Predictions",
))
# Perfect prediction line
min_val = min(y_test.min(), y_pred_chosen.min())
max_val = max(y_test.max(), y_pred_chosen.max())
fig_ap.add_trace(go.Scatter(
    x=[min_val, max_val], y=[min_val, max_val],
    mode="lines", line=dict(color="gray", dash="dash"),
    name="Perfect Prediction",
))
apply_common_layout(fig_ap, title="Actual vs Predicted Temperature", height=450)
fig_ap.update_layout(xaxis_title="Actual Temperature (C)", yaxis_title="Predicted Temperature (C)")
st.plotly_chart(fig_ap, use_container_width=True)

# Residual statistics
resid_stats = pd.DataFrame({
    "Statistic": ["Mean Residual", "Std of Residuals", "Median Residual", "Skewness", "Max Overpredict", "Max Underpredict"],
    "Value": [
        f"{residuals.mean():.4f} C",
        f"{residuals.std():.4f} C",
        f"{np.median(residuals):.4f} C",
        f"{pd.Series(residuals).skew():.4f}",
        f"{residuals.min():.2f} C",
        f"{residuals.max():.2f} C",
    ]
})
st.dataframe(resid_stats, use_container_width=True, hide_index=True)

if abs(residuals.mean()) > 0.5:
    warning_box("The mean residual is notably far from zero -- your model has a systematic bias. It is consistently over- or under-predicting, which means there is a pattern it has failed to learn.")
elif abs(pd.Series(residuals).skew()) > 1.0:
    warning_box("The residuals are skewed, which means the model makes bigger mistakes in one direction than the other. It might be great at predicting mild temperatures but terrible at extremes (or vice versa).")
else:
    st.success("Residuals look well-behaved: centered near zero, roughly symmetric, moderate spread. The model is making honest, random-looking errors -- which is about the best you can hope for.")

# ── 46.5 MAE vs RMSE comparison ─────────────────────────────────────────────
st.header("46.5  MAE vs RMSE: When Does It Matter?")

concept_box(
    "MAE vs RMSE",
    "Here is a surprisingly subtle question: when do MAE and RMSE tell different "
    "stories? RMSE penalizes large errors more than MAE (because it squares them "
    "before averaging). So if RMSE is much bigger than MAE, that means a few "
    "predictions are <b>spectacularly wrong</b>, pulling the RMSE up while the MAE "
    "stays relatively calm. If RMSE and MAE are close, errors are uniform -- nobody "
    "is spectacularly wrong, everyone is just a little bit wrong."
)

ratio_df = results_df[["Model", "MAE (C)", "RMSE (C)"]].copy()
ratio_df["RMSE / MAE Ratio"] = ratio_df["RMSE (C)"] / ratio_df["MAE (C)"]
st.dataframe(ratio_df.style.format({"MAE (C)": "{:.3f}", "RMSE (C)": "{:.3f}", "RMSE / MAE Ratio": "{:.3f}"}),
             use_container_width=True, hide_index=True)

insight_box(
    "An RMSE/MAE ratio close to 1.0 means errors are uniform -- the model is "
    "consistently a little bit off. A ratio above about 1.4 is a red flag: some "
    "predictions are very wrong. When you see that, go investigate the outlier "
    "residuals -- there is probably a systematic pattern the model is missing for "
    "certain types of inputs."
)

code_example("""from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Residuals
residuals = y_test - y_pred
print(f"Mean residual: {residuals.mean():.4f}")
print(f"Std residual:  {residuals.std():.4f}")
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "What does an R-squared of 0.85 mean?",
    [
        "The model has 85% accuracy",
        "The model explains 85% of the variance in the target variable",
        "85% of predictions are within 1 degree of the actual value",
        "The model has 15% error rate",
    ],
    correct_idx=1,
    explanation="R-squared measures the proportion of variance in the target that the model explains. 0.85 means the model captures 85% of what makes temperatures go up and down, and the remaining 15% is unexplained (noise, missing features, or model limitations).",
    key="ch46_quiz1",
)

quiz(
    "RMSE is preferred over MSE because:",
    [
        "RMSE is always smaller than MSE",
        "RMSE is in the same units as the target variable",
        "RMSE ignores outliers",
        "RMSE is faster to compute",
    ],
    correct_idx=1,
    explanation="MSE is in squared units (C-squared, which is not a thing anyone has intuition about). RMSE takes the square root, putting the error back in degrees Celsius -- a unit you can actually reason about in the physical world.",
    key="ch46_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "**MAE** gives you the average error magnitude in original units. It is robust to outliers and easy to explain to non-technical people.",
    "**RMSE** penalizes large errors more heavily. If you care a lot about avoiding big misses, RMSE is your metric.",
    "**R-squared** tells you what fraction of the variance your model explains. 1.0 = perfect, 0.0 = you might as well predict the mean.",
    "Always check **residual plots** -- patterns in residuals are the model trying to tell you something. A random cloud of points around zero is good. A trend or funnel shape is bad.",
    "Physical interpretation matters: RMSE of 3.2 C means predictions are off by roughly 3 degrees. That is the difference between 'nice day for a walk' and 'maybe bring a sweater.'",
    "If RMSE >> MAE, a few predictions are spectacularly wrong -- investigate the outlier residuals to find out what your model is missing.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 45: ROC/AUC & Confusion Matrix",
    prev_page="45_ROC_AUC_Confusion_Matrix.py",
    next_label="Ch 47: Bagging",
    next_page="47_Bagging.py",
)
