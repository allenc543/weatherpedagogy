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
    "In the last chapter, the model predicted a category -- 'this reading is from "
    "Houston' -- and we measured whether it was right or wrong. Now the model is "
    "predicting a *number*: the temperature in degrees Celsius. The question is no "
    "longer 'did you get it right?' but 'how far off were you?'"
)
st.markdown(
    "**The task**: Given a weather reading's humidity, wind speed, surface pressure, "
    "hour of day, and month, predict the temperature. One number out. The model "
    "says 24.3 degrees C and the actual temperature was 27.1 degrees C -- that is an "
    "error of 2.8 degrees. But how do you summarize those errors across thousands of "
    "predictions into a single number that tells you whether the model is any good? "
    "That is what **MSE**, **RMSE**, **MAE**, and **R-squared** are for -- four "
    "different ways of answering the same question, each with a different emphasis."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 46.1 Metric Definitions ─────────────────────────────────────────────────
st.header("46.1  Regression Metric Definitions")

st.markdown(
    "Before I define the formulas, let me set up the intuition. Suppose our model "
    "makes 5 temperature predictions, with errors of +2, -3, +1, -1, and +6 degrees C. "
    "How do we turn those 5 errors into one summary number?"
)

concept_box(
    "Four Ways to Measure 'How Far Off Were You?'",
    "Take those five errors: +2, -3, +1, -1, +6 degrees C.<br><br>"
    "- <b>MAE</b> (Mean Absolute Error): Ignore the sign, average the magnitudes. "
    "(2 + 3 + 1 + 1 + 6) / 5 = 2.6 degrees C. This is the most intuitive -- 'on average, "
    "the model is off by about 2.6 degrees.' A straight answer to a straight question.<br><br>"
    "- <b>MSE</b> (Mean Squared Error): Square each error, then average. "
    "(4 + 9 + 1 + 1 + 36) / 5 = 10.2 degrees-squared. The squaring means that +6 degree "
    "error contributes 36 to the sum, while the +1 degree errors each contribute only 1. "
    "MSE <em>really</em> hates big mistakes.<br><br>"
    "- <b>RMSE</b> (Root Mean Squared Error): Take the square root of MSE. "
    "sqrt(10.2) = 3.2 degrees C. Back in the original units, but still penalizes big "
    "errors more than MAE does. Notice RMSE (3.2) is larger than MAE (2.6) -- that is "
    "always true, and the gap tells you something about outlier errors.<br><br>"
    "- <b>R-squared</b>: 'What fraction of the temperature variation does the model explain?' "
    "If temperatures in our dataset range from -5 to 42 degrees C, and the model captures "
    "most of that variation, R-squared might be 0.85 -- meaning the model explains 85% of "
    "what makes temperatures go up and down. The remaining 15% is unexplained."
)

col1, col2 = st.columns(2)
with col1:
    formula_box(
        "Mean Absolute Error (MAE)",
        r"\underbrace{\text{MAE}}_{\text{avg miss distance}} = \frac{1}{\underbrace{n}_{\text{num predictions}}}\sum_{i=1}^{n}\underbrace{|y_i - \hat{y}_i|}_{\text{absolute error per reading}}",
        "For our weather model: if MAE = 3.0 C, the model's temperature predictions "
        "are typically off by about 3 degrees. That is the difference between 'T-shirt "
        "weather' and 'maybe grab a light jacket.'"
    )
    formula_box(
        "Mean Squared Error (MSE)",
        r"\underbrace{\text{MSE}}_{\text{avg squared miss}} = \frac{1}{\underbrace{n}_{\text{num predictions}}}\sum_{i=1}^{n}\underbrace{(y_i - \hat{y}_i)^2}_{\text{squared error per reading}}",
        "A prediction off by 10 degrees contributes 100 to the MSE; a prediction off "
        "by 1 degree contributes only 1. If you care about avoiding catastrophic misses "
        "-- like predicting 20 C when it is actually 35 C -- MSE is the metric that "
        "shares your priorities."
    )
with col2:
    formula_box(
        "Root Mean Squared Error (RMSE)",
        r"\underbrace{\text{RMSE}}_{\text{typical error in degrees}} = \sqrt{\underbrace{\text{MSE}}_{\text{mean squared error}}}",
        "RMSE of 4.2 C means your typical error is about 4 degrees, in units you can "
        "feel. The difference between RMSE and MAE tells you about outliers: if RMSE is "
        "much bigger than MAE, a few predictions are spectacularly wrong."
    )
    formula_box(
        "R-squared (Coefficient of Determination)",
        r"\underbrace{R^2}_{\text{variance explained}} = 1 - \frac{\underbrace{\sum(y_i - \hat{y}_i)^2}_{\text{model's leftover error}}}{\underbrace{\sum(y_i - \bar{y})^2}_{\text{total temp variation}}}",
        "R-squared compares your model to the dumbest possible model: always predicting "
        "the average temperature. If R-squared = 0.0, your fancy model does no better than "
        "just saying 'it is probably 20 C' every time. R-squared = 1.0 means perfection. "
        "It can actually go negative if your model is somehow worse than the mean."
    )

# ── 46.2 Compare Models ─────────────────────────────────────────────────────
st.header("46.2  Compare Metrics Across Regression Models")

st.markdown(
    "Let us make this concrete. We will predict **temperature** from humidity, wind "
    "speed, surface pressure, hour of day, and month. We will train 6 different "
    "regression models and compare their metrics side by side. The question is: which "
    "model produces temperature predictions closest to reality, and by how much?"
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

st.markdown(
    "Look at the table above. Linear regression, Ridge, and Lasso all produce very "
    "similar numbers -- that is because our features have a mostly linear relationship "
    "with temperature, so regularization does not buy much here. The tree-based models "
    "(Random Forest, Gradient Boosting) often do better because they can capture "
    "non-linear patterns, like the fact that humidity affects temperature differently "
    "at different times of day."
)

# ── 46.3 Physical Interpretation ────────────────────────────────────────────
st.header("46.3  Physical Interpretation of Metrics")

best_model_name = results_df.iloc[0]["Model"]
best_rmse = results_df.iloc[0]["RMSE (C)"]
best_mae = results_df.iloc[0]["MAE (C)"]

st.markdown(f"""
The best model (**{best_model_name}**) achieves:
- **RMSE = {best_rmse:.2f} C** -- on average, predictions are off by about {best_rmse:.1f} degrees Celsius.
- **MAE = {best_mae:.2f} C** -- the typical prediction misses by about {best_mae:.1f} degrees.

But what does 'off by {best_rmse:.1f} degrees' actually *feel* like? An error of 3 degrees C is the difference between 22 C (comfortable in a T-shirt) and 25 C (starting to feel warm). An error of 5 degrees is the difference between 'nice spring day' and 'turn on the air conditioning.' Let us put these numbers in context by looking at the temperature ranges in our data:
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
    "Dallas temperature ranges from roughly -5 C to 42 C -- a span of 47 degrees. "
    f"An error of {best_rmse:.1f} C is a small fraction of that range. You would trust "
    "this model for deciding whether to wear a jacket, but not for calibrating a "
    "precision thermometer. This is the kind of physical reasoning you should always "
    "apply: an RMSE of 3 means nothing in the abstract, but 'off by about 3 degrees "
    "on a day that could be anywhere from freezing to scorching' is actually quite good."
)

# ── 46.4 Residual Plots ─────────────────────────────────────────────────────
st.header("46.4  Residual Analysis")

st.markdown(
    "A single number like RMSE tells you the *size* of the model's errors but not "
    "their *pattern*. Residual analysis is how you figure out *where* and *why* the "
    "model goes wrong."
)

concept_box(
    "Residuals: The Model's Mistakes, Laid Bare",
    "For each prediction, the residual is simply: Actual Temperature - Predicted "
    "Temperature. If the model predicted 25 C and the actual was 28 C, the residual "
    "is +3 C (the model undershot). If it predicted 30 C and the actual was 27 C, "
    "the residual is -3 C (the model overshot).<br><br>"
    "What you <em>want</em> to see in a good model's residuals:<br>"
    "- <b>Randomly scattered around zero</b>: no pattern. If you see the residuals "
    "forming a curve or a funnel, the model is systematically wrong at certain "
    "temperature ranges and there is a pattern it could still learn.<br>"
    "- <b>Normally distributed</b>: the histogram of residuals should look like a "
    "bell curve centered at zero. The model is equally likely to overpredict as "
    "underpredict, and by similar amounts.<br>"
    "- <b>Constant spread</b>: the model should not be off by 1 degree at low "
    "temperatures and off by 8 degrees at high temperatures. If the spread fans out, "
    "that is called heteroscedasticity, and it means the model is much worse at "
    "predicting some temperature ranges than others."
)

model_choice = st.selectbox("Model for residual analysis", list(trained_models.keys()), key="resid_model")
_, y_pred_chosen = trained_models[model_choice]
residuals = y_test.values - y_pred_chosen

st.markdown(
    f"Below are the residual plots for **{model_choice}**. The left plot shows "
    f"residuals vs predicted temperature -- look for patterns. The right plot shows "
    f"the histogram of residuals -- look for symmetry around zero."
)

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
st.markdown(
    "This next plot is the most intuitive: actual temperature (x-axis) vs predicted "
    "temperature (y-axis). If the model were perfect, every point would land on the "
    "dashed diagonal line. Points above the line mean the model overpredicted; points "
    "below mean it underpredicted. How tightly the cloud hugs the line tells you how "
    "good the model is."
)

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
    warning_box(
        f"The mean residual is {residuals.mean():.2f} C -- notably far from zero. "
        "This means the model has a systematic bias: it is consistently over- or "
        "under-predicting temperatures. For example, if the mean residual is +2 C, "
        "the model is consistently underestimating temperatures by about 2 degrees. "
        "This suggests there is a pattern the model has not learned -- perhaps it is "
        "missing a feature like solar radiation or cloud cover that would help."
    )
elif abs(pd.Series(residuals).skew()) > 1.0:
    warning_box(
        "The residuals are skewed, meaning the model makes bigger mistakes in one "
        "direction. It might be accurate for mild temperatures (15-25 C) but "
        "systematically underpredict during heat waves or overpredict during cold "
        "snaps. If you see this, investigate: plot residuals vs time of year and "
        "you may find the model struggles with extreme temperatures."
    )
else:
    st.success(
        "Residuals look well-behaved: centered near zero, roughly symmetric, "
        "moderate spread. The model is making honest, random-looking errors -- no "
        "systematic pattern that a more complex model could exploit. This is about "
        "the best you can hope for."
    )

# ── 46.5 MAE vs RMSE comparison ─────────────────────────────────────────────
st.header("46.5  MAE vs RMSE: When Does It Matter?")

st.markdown(
    "Here is a question that trips up a lot of people: if MAE and RMSE both measure "
    "'how far off are the predictions,' why do we need both? The answer lies in how "
    "they treat outlier errors."
)

concept_box(
    "MAE vs RMSE: A Weather Example",
    "Suppose your model makes 100 temperature predictions. 99 of them are off by 1 "
    "degree C, and one is off by 20 degrees C (maybe it wildly misjudged a freak cold "
    "front).<br><br>"
    "<b>MAE</b> = (99 * 1 + 1 * 20) / 100 = <b>1.19 C</b> -- the one big miss barely "
    "moves the average, because MAE treats every error equally.<br><br>"
    "<b>RMSE</b> = sqrt((99 * 1 + 1 * 400) / 100) = sqrt(4.99) = <b>2.23 C</b> -- "
    "almost double the MAE, because squaring that 20-degree error turns it into 400, "
    "which dominates the sum.<br><br>"
    "So if RMSE is much bigger than MAE, it is a signal that a few predictions are "
    "<b>spectacularly wrong</b>. If they are close, errors are uniform -- nothing is "
    "spectacularly wrong, everything is just a little bit off."
)

ratio_df = results_df[["Model", "MAE (C)", "RMSE (C)"]].copy()
ratio_df["RMSE / MAE Ratio"] = ratio_df["RMSE (C)"] / ratio_df["MAE (C)"]
st.dataframe(ratio_df.style.format({"MAE (C)": "{:.3f}", "RMSE (C)": "{:.3f}", "RMSE / MAE Ratio": "{:.3f}"}),
             use_container_width=True, hide_index=True)

insight_box(
    "An RMSE/MAE ratio close to 1.0 means the model's errors are uniform -- it is "
    "off by a consistent amount for every prediction. A ratio above about 1.4 is a "
    "red flag: some predictions are wildly wrong while most are fine. In weather terms, "
    "the model might nail the temperature on 95% of days but completely botch it during "
    "unusual weather events -- sudden cold fronts, heat waves, or temperature inversions. "
    "When you see a high ratio, go find those outlier predictions and figure out what "
    "makes them different."
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
    "Your temperature prediction model has R-squared = 0.85. What does this mean?",
    [
        "The model has 85% accuracy",
        "The model explains 85% of the variance in temperature",
        "85% of predictions are within 1 degree of the actual value",
        "The model has a 15% error rate",
    ],
    correct_idx=1,
    explanation=(
        "R-squared = 0.85 means the model captures 85% of what makes temperatures "
        "vary across our dataset. Temperature varies because of time of year, time of "
        "day, weather systems, and location. The model explains 85% of that variation "
        "using humidity, wind speed, pressure, hour, and month. The remaining 15% is "
        "unexplained -- maybe it is cloud cover, precipitation, or just random "
        "day-to-day weather noise that our features do not capture."
    ),
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
    explanation=(
        "MSE is in squared units -- degrees-Celsius-squared, which is not a unit anyone "
        "has physical intuition about. An MSE of 16 sounds bad, but is it? You cannot "
        "tell. RMSE takes the square root: sqrt(16) = 4 degrees C. Now you can reason "
        "about it: 'the model's typical error is about 4 degrees -- that is the difference "
        "between a warm spring day and a mildly cool one.' RMSE puts the error back in "
        "units you can feel."
    ),
    key="ch46_quiz2",
)

quiz(
    "Your model has MAE = 2.1 C and RMSE = 5.8 C. What does the large gap tell you?",
    [
        "The model is overfitting",
        "The model is underfitting",
        "A few predictions are extremely wrong, even though most are close",
        "The model needs more training data",
    ],
    correct_idx=2,
    explanation=(
        "An RMSE/MAE ratio of 5.8 / 2.1 = 2.76 is very high. Most predictions are off "
        "by only about 2 degrees (the MAE), but a handful are off by 15 or 20 degrees, "
        "and the squaring in RMSE amplifies those outliers enormously. This pattern is "
        "common in weather prediction during extreme events -- the model handles typical "
        "days well but completely misses heat waves or cold fronts. The fix: investigate "
        "which predictions have the largest residuals and see if they share a pattern "
        "(all nighttime? all during summer? all from one city?)."
    ),
    key="ch46_quiz3",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "**MAE** is the average miss distance in original units. If MAE = 3.0 C, the model's temperature predictions are typically off by about 3 degrees. It is robust to outliers and the easiest metric to explain.",
    "**RMSE** penalizes large errors more heavily because of the squaring. If you care about avoiding catastrophic misses -- predicting 20 C when it is actually 35 C -- RMSE is your metric.",
    "**R-squared** tells you what fraction of temperature variation the model explains. R-squared = 0.85 means the model captures 85% of what makes temperatures go up and down; the other 15% is noise or missing features.",
    "Always check **residual plots**. A random cloud of points centered at zero is good. A curve, funnel, or systematic lean means the model is missing a pattern it could learn.",
    "Physical interpretation matters: RMSE of 3.2 C means 'off by roughly 3 degrees.' That is the difference between 'nice day for a walk' and 'maybe bring a sweater.' Always translate metrics into units you can feel.",
    "If RMSE >> MAE, a few predictions are spectacularly wrong. Find those outlier residuals -- they often correspond to extreme weather events the model does not handle well.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 45: ROC/AUC & Confusion Matrix",
    prev_page="45_ROC_AUC_Confusion_Matrix.py",
    next_label="Ch 47: Bagging",
    next_page="47_Bagging.py",
)
