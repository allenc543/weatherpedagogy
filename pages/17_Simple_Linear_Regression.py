"""Chapter 17: Simple Linear Regression — OLS, R-squared, residuals, assumptions."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, color_map, scatter_chart
from utils.ml_helpers import prepare_regression_data, regression_metrics
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)
from utils.constants import CITY_LIST, FEATURE_COLS, FEATURE_LABELS, CITY_COLORS

# ---------------------------------------------------------------------------
st.set_page_config(page_title="Ch 17: Simple Linear Regression", layout="wide")
df = load_data()
fdf = sidebar_filters(df)

chapter_header(17, "Simple Linear Regression", part="IV")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "Simple Linear Regression: Drawing the Best Straight Line Through Data",
    "Here's the basic idea. You have a bunch of points on a scatter plot, and you want to draw "
    "a straight line through them. Not just any line -- the <em>best</em> line, the one that comes "
    "closest to all the points simultaneously. This is <b>Ordinary Least Squares (OLS)</b>, and "
    "'closest' means 'minimizes the sum of squared vertical distances from the points to the line.' "
    "Why squared? Because if you just summed the raw distances, positive and negative errors would "
    "cancel out, and a terrible line that passes through the middle of nowhere could score well. "
    "Squaring forces all errors to be positive and penalizes big misses disproportionately.",
)

col1, col2 = st.columns(2)
with col1:
    formula_box(
        "The Linear Model",
        r"\hat{y} = \beta_0 + \beta_1 x",
        "beta_0 is the intercept (the predicted y when x = 0, which may or may not be meaningful depending on your data). beta_1 is the slope -- for every 1-unit increase in x, y changes by beta_1 units. That's the number everyone cares about.",
    )
with col2:
    formula_box(
        "R-squared (Coefficient of Determination)",
        r"R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}",
        "What fraction of y's variability does the model explain? R-squared = 0 means the model is no better than just predicting the mean every time. R-squared = 1 means the model captures everything. In practice, you live somewhere in between and argue about whether 0.6 is 'good enough.'",
    )

st.markdown("### The Four Assumptions You're Making (Whether You Know It or Not)")
a1, a2, a3, a4 = st.columns(4)
with a1:
    st.markdown("**1. Linearity**")
    st.caption("The relationship between X and Y is actually a straight line. If it's curved, your line will systematically miss.")
with a2:
    st.markdown("**2. Independence**")
    st.caption("Each observation is its own thing, not influenced by the others. Time series data often violates this.")
with a3:
    st.markdown("**3. Homoscedasticity**")
    st.caption("The variance of residuals is constant across all values of X. Fancy word for 'the spread doesn't change.'")
with a4:
    st.markdown("**4. Normality of Residuals**")
    st.caption("The residuals (errors) follow a normal distribution. This matters most for hypothesis tests on the coefficients.")

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Fit a Regression Line
# ---------------------------------------------------------------------------
st.subheader("Try It: Fit a Regression Line to Real Weather Data")

ctrl_col, plot_col = st.columns([1, 2])

with ctrl_col:
    reg_city = st.selectbox("City", CITY_LIST, key="reg_city")
    predictor = st.selectbox(
        "Predictor (X)",
        FEATURE_COLS,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
        index=1,  # humidity
        key="reg_x",
    )
    target = st.selectbox(
        "Target (Y)",
        FEATURE_COLS,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
        index=0,  # temperature
        key="reg_y",
    )

city_data = fdf[fdf["city"] == reg_city][[predictor, target]].dropna()

if len(city_data) > 10:
    X = city_data[predictor].values.reshape(-1, 1)
    y = city_data[target].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)
    residuals = y - y_pred

    # Sample for scatter plot
    if len(city_data) > 5000:
        sample_idx = np.random.RandomState(42).choice(len(city_data), 5000, replace=False)
    else:
        sample_idx = np.arange(len(city_data))

    with plot_col:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X[sample_idx].flatten(), y=y[sample_idx],
            mode="markers", name="Data",
            marker=dict(color=CITY_COLORS.get(reg_city, "#636EFA"), size=3, opacity=0.3),
        ))
        x_line = np.array([X.min(), X.max()])
        fig.add_trace(go.Scatter(
            x=x_line.flatten(), y=model.predict(x_line.reshape(-1, 1)),
            mode="lines", name=f"OLS: y = {slope:.3f}x + {intercept:.2f}",
            line=dict(color="#E63946", width=3),
        ))
        apply_common_layout(fig, title=f"Simple Linear Regression: {FEATURE_LABELS[predictor]} -> {FEATURE_LABELS[target]}")
        fig.update_xaxes(title_text=FEATURE_LABELS[predictor])
        fig.update_yaxes(title_text=FEATURE_LABELS[target])
        st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Slope (beta_1)", f"{slope:.4f}")
    m2.metric("Intercept (beta_0)", f"{intercept:.2f}")
    m3.metric("R-squared", f"{r_squared:.4f}")

    st.markdown(
        f"**What this means in plain English:** For every 1-unit increase in {FEATURE_LABELS[predictor]}, "
        f"{FEATURE_LABELS[target]} changes by {slope:.4f} units on average. "
        f"The model explains {r_squared * 100:.1f}% of the variance in {FEATURE_LABELS[target]}. "
        f"{'That is pretty good.' if r_squared > 0.5 else 'There is a lot of unexplained variability -- other factors matter too.'}"
    )

    st.divider()

    # -------------------------------------------------------------------
    # 3. Manual Slope/Intercept Slider (Least Squares Intuition)
    # -------------------------------------------------------------------
    st.subheader("Build Your Intuition: Try to Beat the Algorithm")

    st.markdown(
        "Here's a fun exercise. Adjust the slope and intercept manually and watch the "
        "Sum of Squared Residuals (SSR) change. Your goal: get as close to the OLS solution "
        "as possible. The OLS algorithm finds the mathematically optimal line in one shot -- "
        "but trying to do it by hand gives you a visceral sense of what 'minimizing squared errors' "
        "actually means."
    )

    manual_col1, manual_col2 = st.columns([1, 2])

    with manual_col1:
        manual_slope = st.slider(
            "Slope", float(slope - 2), float(slope + 2), float(slope),
            step=0.01, key="manual_slope",
        )
        manual_intercept = st.slider(
            "Intercept", float(intercept - 20), float(intercept + 20), float(intercept),
            step=0.1, key="manual_intercept",
        )

    manual_pred = manual_slope * X.flatten() + manual_intercept
    manual_ssr = np.sum((y - manual_pred) ** 2)
    ols_ssr = np.sum(residuals ** 2)

    with manual_col2:
        # Use a smaller sample for the manual fit
        small_idx = np.random.RandomState(0).choice(len(city_data), min(500, len(city_data)), replace=False)
        fig_manual = go.Figure()
        fig_manual.add_trace(go.Scatter(
            x=X[small_idx].flatten(), y=y[small_idx],
            mode="markers", name="Data",
            marker=dict(color=CITY_COLORS.get(reg_city, "#636EFA"), size=5, opacity=0.5),
        ))
        # Manual line
        fig_manual.add_trace(go.Scatter(
            x=x_line.flatten(),
            y=manual_slope * x_line.flatten() + manual_intercept,
            mode="lines", name=f"Your line (SSR={manual_ssr:.0f})",
            line=dict(color="#F4A261", width=3, dash="solid"),
        ))
        # OLS line
        fig_manual.add_trace(go.Scatter(
            x=x_line.flatten(),
            y=model.predict(x_line.reshape(-1, 1)),
            mode="lines", name=f"OLS (SSR={ols_ssr:.0f})",
            line=dict(color="#E63946", width=2, dash="dash"),
        ))
        apply_common_layout(fig_manual, title="Manual vs OLS Fit")
        fig_manual.update_xaxes(title_text=FEATURE_LABELS[predictor])
        fig_manual.update_yaxes(title_text=FEATURE_LABELS[target])
        st.plotly_chart(fig_manual, use_container_width=True)

    ssr1, ssr2 = st.columns(2)
    ssr1.metric("Your SSR", f"{manual_ssr:,.0f}")
    ssr2.metric("OLS SSR (minimum)", f"{ols_ssr:,.0f}",
                delta=f"{manual_ssr - ols_ssr:+,.0f}")

    if abs(manual_ssr - ols_ssr) < ols_ssr * 0.01:
        st.success("Your line is within 1% of the OLS solution. Nicely done.")
    else:
        st.info("Keep tweaking. The OLS line (dashed red) is the mathematically optimal solution -- it's the global minimum of the SSR surface.")

    st.divider()

    # -------------------------------------------------------------------
    # 4. Residual Diagnostics
    # -------------------------------------------------------------------
    st.subheader("Reading the Residuals: Where the Model Fails")

    st.markdown(
        "The residuals (actual minus predicted) are where all the secrets hide. If the model is "
        "doing its job, residuals should look like random noise: scattered evenly around zero, "
        "with no patterns. If you see patterns, something is wrong -- maybe the relationship "
        "isn't linear, maybe the variance isn't constant, maybe something else entirely."
    )

    diag_col1, diag_col2 = st.columns(2)

    with diag_col1:
        # Residuals vs Fitted
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Scatter(
            x=y_pred[sample_idx], y=residuals[sample_idx],
            mode="markers", marker=dict(color="#2A9D8F", size=3, opacity=0.3),
            showlegend=False,
        ))
        fig_resid.add_hline(y=0, line_dash="dash", line_color="#E63946")
        apply_common_layout(fig_resid, title="Residuals vs Fitted Values")
        fig_resid.update_xaxes(title_text="Fitted Values")
        fig_resid.update_yaxes(title_text="Residuals")
        st.plotly_chart(fig_resid, use_container_width=True)

    with diag_col2:
        # Residual histogram
        fig_rhist = go.Figure()
        fig_rhist.add_trace(go.Histogram(
            x=residuals, nbinsx=60, marker_color="#7209B7", opacity=0.7,
        ))
        apply_common_layout(fig_rhist, title="Distribution of Residuals")
        fig_rhist.update_xaxes(title_text="Residual")
        fig_rhist.update_yaxes(title_text="Count")
        st.plotly_chart(fig_rhist, use_container_width=True)

    # Q-Q plot
    fig_qq = go.Figure()
    sorted_resid = np.sort(residuals)
    theoretical_q = sp_stats.norm.ppf(np.linspace(0.001, 0.999, len(sorted_resid)))
    fig_qq.add_trace(go.Scatter(
        x=theoretical_q[::max(1, len(sorted_resid) // 2000)],
        y=sorted_resid[::max(1, len(sorted_resid) // 2000)],
        mode="markers", marker=dict(color="#2A9D8F", size=3, opacity=0.5),
        showlegend=False,
    ))
    q_min, q_max = theoretical_q.min(), theoretical_q.max()
    fig_qq.add_trace(go.Scatter(
        x=[q_min, q_max], y=[q_min * residuals.std() + residuals.mean(),
                              q_max * residuals.std() + residuals.mean()],
        mode="lines", line=dict(color="#E63946", dash="dash"), showlegend=False,
    ))
    apply_common_layout(fig_qq, title="Q-Q Plot of Residuals", height=400)
    fig_qq.update_xaxes(title_text="Theoretical Quantiles")
    fig_qq.update_yaxes(title_text="Sample Quantiles")
    st.plotly_chart(fig_qq, use_container_width=True)

    insight_box(
        "Three things to look for: (1) The residuals-vs-fitted plot should show no pattern -- "
        "just a random cloud. A U-shape means the relationship is curved; a funnel shape means "
        "the variance isn't constant. (2) The histogram should be roughly bell-shaped. "
        "(3) The Q-Q plot should follow the diagonal. Deviations at the tails mean the residuals "
        "have heavier or lighter tails than a normal distribution."
    )

else:
    st.warning("Not enough data for selected city. Adjust sidebar filters.")

st.divider()

# ---------------------------------------------------------------------------
# 5. Code Example
# ---------------------------------------------------------------------------
code_example("""
from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare data
X = df[['relative_humidity_pct']].values
y = df['temperature_c'].values

# Fit model
model = LinearRegression()
model.fit(X, y)

# Results
print(f"Slope: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"R-squared: {model.score(X, y):.4f}")

# Residual analysis
y_pred = model.predict(X)
residuals = y - y_pred
print(f"Mean residual: {residuals.mean():.6f}")  # should be ~0
print(f"Std residual: {residuals.std():.2f}")

# Prediction
new_humidity = np.array([[65]])
predicted_temp = model.predict(new_humidity)
print(f"Predicted temp at 65% humidity: {predicted_temp[0]:.2f} C")
""")

st.divider()

# ---------------------------------------------------------------------------
# 6. Quiz
# ---------------------------------------------------------------------------
quiz(
    "In simple linear regression, R-squared = 0.64 means:",
    [
        "The correlation is 0.64",
        "64% of the variance in Y is explained by X",
        "The model is 64% accurate",
        "The slope is 0.64",
    ],
    correct_idx=1,
    explanation="R-squared tells you what fraction of Y's variability your model captures. 0.64 means 64% is explained, 36% is left over as residual variance. (Fun fact: for simple regression, R-squared equals the correlation squared. So the correlation here is sqrt(0.64) = 0.8.)",
    key="ch17_quiz1",
)

quiz(
    "If residuals show a funnel shape (increasing spread) in the residuals-vs-fitted plot, which assumption is violated?",
    [
        "Linearity",
        "Independence",
        "Homoscedasticity",
        "Normality of residuals",
    ],
    correct_idx=2,
    explanation="A funnel shape means the residual variance changes across the range of fitted values -- the spread isn't constant. This is heteroscedasticity, the evil twin of homoscedasticity. It doesn't bias your coefficient estimates, but it does mess up your standard errors and therefore your p-values.",
    key="ch17_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 7. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Simple linear regression draws the best straight line through data using Ordinary Least Squares, which minimizes the sum of squared vertical distances from points to line.",
    "R-squared tells you what fraction of Y's variability the model explains. An R-squared of 0 means 'I could have just guessed the mean.' An R-squared of 1 means 'I nailed every single point.'",
    "The slope is the number everyone cares about: it tells you how much Y changes, on average, when X increases by one unit. The intercept is where the line hits the Y-axis.",
    "Residual plots are your diagnostic toolkit. No pattern = good. Curves = non-linearity. Funnels = heteroscedasticity. Heavy tails on the Q-Q plot = non-normal residuals.",
    "OLS makes four assumptions (linearity, independence, constant variance, normal residuals). Violating them doesn't make OLS useless, but it changes how much you should trust the p-values and confidence intervals.",
])

navigation(
    prev_label="Ch 16: Correlation Analysis",
    prev_page="16_Correlation_Analysis.py",
    next_label="Ch 18: Multiple Regression",
    next_page="18_Multiple_Regression.py",
)
