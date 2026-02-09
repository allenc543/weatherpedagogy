"""Chapter 19: Polynomial Regression — Overfitting vs underfitting, degree selection."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, color_map
from utils.ml_helpers import regression_metrics
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)
from utils.constants import CITY_LIST, FEATURE_LABELS, CITY_COLORS

# ---------------------------------------------------------------------------
df = load_data()
fdf = sidebar_filters(df)

chapter_header(19, "Polynomial Regression", part="IV")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "When Straight Lines Aren't Enough",
    "So far we've been fitting straight lines to data, which is great when the relationship "
    "is actually linear. But temperature over the course of a year? That's a sinusoidal curve. "
    "No straight line in the world will capture it. <b>Polynomial regression</b> is the natural "
    "next step: add x-squared, x-cubed, and higher powers to let the model bend. The weird and "
    "wonderful thing is that the model is still 'linear' in the statistical sense -- it's linear "
    "in the <em>parameters</em> (the betas). The curve might wiggle, but the math is still just "
    "adding up weighted terms.",
)

formula_box(
    "Polynomial Regression Model (degree d)",
    r"\underbrace{\hat{y}}_{\text{predicted temp}} = \underbrace{\beta_0}_{\text{intercept}} + \underbrace{\beta_1}_{\text{linear coeff}} \underbrace{x}_{\text{day of year}} + \underbrace{\beta_2}_{\text{quadratic coeff}} x^2 + \cdots + \underbrace{\beta_d}_{\text{degree-d coeff}} x^{\underbrace{d}_{\text{poly degree}}}",
    "Degree 1 = straight line (what we've been doing). Degree 2 = parabola. Degree 3 = cubic. The higher the degree, the more flexible the curve -- but flexibility is a double-edged sword.",
)

st.markdown("### The Bias-Variance Tradeoff: The Central Drama of Machine Learning")
col1, col2, col3 = st.columns(3)
with col1:
    concept_box("Underfitting (High Bias)",
                "Your model is too simple for the pattern in the data. It's like trying to "
                "describe a roller coaster with a straight line. Both training and test errors "
                "are high because the model can't capture what's actually going on.")
with col2:
    concept_box("The Sweet Spot",
                "Your model captures the real pattern without memorizing the noise. Training "
                "error is reasonably low, test error is close to training error. This is the "
                "Goldilocks zone -- complex enough to be useful, simple enough to generalize.")
with col3:
    concept_box("Overfitting (High Variance)",
                "Your model is too complex. It has memorized not just the signal but also the "
                "noise in the training data. Training error is fantastic; test error is terrible. "
                "It's like a student who memorized the answer key instead of learning the material.")

warning_box(
    "A degree-15 polynomial can thread its way through every single training point like a "
    "needle through fabric. It will have near-perfect training accuracy. And it will be "
    "absolutely terrible at predicting new data, because it's fitting to noise, not signal. "
    "ALWAYS evaluate on held-out test data. Your model's performance on data it's already seen "
    "tells you nothing about its real-world usefulness."
)

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Day-of-Year -> Temperature (Annual Cycle)
# ---------------------------------------------------------------------------
st.subheader("The Playground: Fitting the Annual Temperature Curve")

st.markdown(
    "Temperature over the year follows an approximately sinusoidal pattern -- warm in summer, "
    "cool in winter, smooth transitions in between. This makes day-of-year vs temperature "
    "the perfect testing ground for polynomial regression. Use the slider to crank up the "
    "degree and watch the model go from 'too simple' to 'just right' to 'dangerously overfit.'"
)

poly_city = st.selectbox("City", CITY_LIST, key="poly_city")
city_data = fdf[fdf["city"] == poly_city][["day_of_year", "temperature_c"]].dropna()

if len(city_data) > 100:
    X_all = city_data["day_of_year"].values.reshape(-1, 1)
    y_all = city_data["temperature_c"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    # Degree slider
    degree = st.slider("Polynomial degree", 1, 15, 3, key="poly_degree")

    # Fit polynomial model
    poly_model = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
    poly_model.fit(X_train, y_train)

    train_pred = poly_model.predict(X_train)
    test_pred = poly_model.predict(X_test)

    train_metrics = regression_metrics(y_train, train_pred)
    test_metrics = regression_metrics(y_test, test_pred)

    # Scatter + fitted curve
    x_curve = np.linspace(1, 366, 500).reshape(-1, 1)
    y_curve = poly_model.predict(x_curve)

    # Sample for plotting
    if len(X_train) > 3000:
        plot_idx_tr = np.random.RandomState(42).choice(len(X_train), 3000, replace=False)
    else:
        plot_idx_tr = np.arange(len(X_train))

    if len(X_test) > 1000:
        plot_idx_te = np.random.RandomState(42).choice(len(X_test), 1000, replace=False)
    else:
        plot_idx_te = np.arange(len(X_test))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X_train[plot_idx_tr].flatten(), y=y_train[plot_idx_tr],
        mode="markers", name="Train",
        marker=dict(color="#2A9D8F", size=3, opacity=0.2),
    ))
    fig.add_trace(go.Scatter(
        x=X_test[plot_idx_te].flatten(), y=y_test[plot_idx_te],
        mode="markers", name="Test",
        marker=dict(color="#F4A261", size=3, opacity=0.3),
    ))
    fig.add_trace(go.Scatter(
        x=x_curve.flatten(), y=y_curve,
        mode="lines", name=f"Degree {degree} Polynomial",
        line=dict(color="#E63946", width=3),
    ))
    apply_common_layout(fig, title=f"Polynomial Regression (degree={degree}): {poly_city}")
    fig.update_xaxes(title_text="Day of Year")
    fig.update_yaxes(title_text="Temperature (C)")
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Train R-squared", f"{train_metrics['r2']:.4f}")
    m2.metric("Test R-squared", f"{test_metrics['r2']:.4f}")
    m3.metric("Train RMSE", f"{train_metrics['rmse']:.2f}")
    m4.metric("Test RMSE", f"{test_metrics['rmse']:.2f}")

    gap = train_metrics["r2"] - test_metrics["r2"]
    if degree <= 2:
        st.info(f"Degree {degree} is likely **underfitting**. A straight line (or parabola) just can't capture a seasonal temperature curve. The model is too rigid.")
    elif gap > 0.1:
        st.error(f"Degree {degree} is likely **overfitting**. The model has memorized the training data (R-squared = {train_metrics['r2']:.4f}) but fails on new data (test R-squared = {test_metrics['r2']:.4f}). That gap is the telltale sign.")
    else:
        st.success(f"Degree {degree} looks like a **good fit**. Train and test metrics are close, which means the model is capturing the real pattern without memorizing noise.")

    st.divider()

    # -------------------------------------------------------------------
    # 3. Training vs Test Error Across Degrees
    # -------------------------------------------------------------------
    st.subheader("The Classic U-Shaped Curve: Error vs Complexity")

    st.markdown(
        "This is arguably the most important plot in all of machine learning. Training error "
        "monotonically decreases as you add complexity (higher degree = more flexibility = "
        "better fit to training data). But test error follows a **U-shape**: it drops as you "
        "add useful complexity, hits a minimum at the sweet spot, then rises as overfitting "
        "kicks in. Your job is to find the bottom of that U."
    )

    degrees = list(range(1, 16))
    train_rmses = []
    test_rmses = []
    train_r2s = []
    test_r2s = []

    for d in degrees:
        pm = make_pipeline(PolynomialFeatures(d, include_bias=False), LinearRegression())
        pm.fit(X_train, y_train)
        tr_pred = pm.predict(X_train)
        te_pred = pm.predict(X_test)
        tr_m = regression_metrics(y_train, tr_pred)
        te_m = regression_metrics(y_test, te_pred)
        train_rmses.append(tr_m["rmse"])
        test_rmses.append(te_m["rmse"])
        train_r2s.append(tr_m["r2"])
        test_r2s.append(te_m["r2"])

    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(
        x=degrees, y=train_rmses, mode="lines+markers", name="Train RMSE",
        line=dict(color="#2A9D8F", width=2), marker=dict(size=6),
    ))
    fig_err.add_trace(go.Scatter(
        x=degrees, y=test_rmses, mode="lines+markers", name="Test RMSE",
        line=dict(color="#E63946", width=2), marker=dict(size=6),
    ))
    fig_err.add_vline(x=degree, line_dash="dash", line_color="#F4A261",
                      annotation_text=f"Your choice: {degree}")
    apply_common_layout(fig_err, title="RMSE vs Polynomial Degree (Bias-Variance Tradeoff)")
    fig_err.update_xaxes(title_text="Polynomial Degree", dtick=1)
    fig_err.update_yaxes(title_text="RMSE (C)")
    st.plotly_chart(fig_err, use_container_width=True)

    # Also show R-squared
    fig_r2 = go.Figure()
    fig_r2.add_trace(go.Scatter(
        x=degrees, y=train_r2s, mode="lines+markers", name="Train R-squared",
        line=dict(color="#2A9D8F", width=2), marker=dict(size=6),
    ))
    fig_r2.add_trace(go.Scatter(
        x=degrees, y=test_r2s, mode="lines+markers", name="Test R-squared",
        line=dict(color="#E63946", width=2), marker=dict(size=6),
    ))
    fig_r2.add_vline(x=degree, line_dash="dash", line_color="#F4A261",
                      annotation_text=f"Your choice: {degree}")
    apply_common_layout(fig_r2, title="R-squared vs Polynomial Degree")
    fig_r2.update_xaxes(title_text="Polynomial Degree", dtick=1)
    fig_r2.update_yaxes(title_text="R-squared")
    st.plotly_chart(fig_r2, use_container_width=True)

    # Find best degree
    best_degree = degrees[np.argmin(test_rmses)]
    insight_box(
        f"The best test RMSE occurs at degree {best_degree} ({test_rmses[best_degree - 1]:.2f} C). "
        f"For the annual temperature cycle, a low-degree polynomial (3-5) usually does the job "
        f"beautifully -- it's complex enough to capture the seasonal curve but simple enough not to "
        f"hallucinate patterns in day-to-day weather noise."
    )

    st.divider()

    # -------------------------------------------------------------------
    # 4. Degree Comparison: Side by Side
    # -------------------------------------------------------------------
    st.subheader("Seeing Is Believing: Degree 1 vs 4 vs 15")

    fig_cmp = go.Figure()
    # Data points (test only for cleanliness)
    fig_cmp.add_trace(go.Scatter(
        x=X_test[plot_idx_te].flatten(), y=y_test[plot_idx_te],
        mode="markers", name="Test Data",
        marker=dict(color="gray", size=3, opacity=0.3),
    ))

    cmp_colors = {1: "#264653", 4: "#2A9D8F", 15: "#E63946"}
    cmp_dashes = {1: "dot", 4: "solid", 15: "dash"}
    for d in [1, 4, 15]:
        pm = make_pipeline(PolynomialFeatures(d, include_bias=False), LinearRegression())
        pm.fit(X_train, y_train)
        y_c = pm.predict(x_curve)
        te_r2 = regression_metrics(y_test, pm.predict(X_test))["r2"]
        fig_cmp.add_trace(go.Scatter(
            x=x_curve.flatten(), y=y_c,
            mode="lines", name=f"Degree {d} (test R2={te_r2:.3f})",
            line=dict(color=cmp_colors[d], width=2, dash=cmp_dashes[d]),
        ))

    apply_common_layout(fig_cmp, title="Underfitting (d=1) vs Good Fit (d=4) vs Overfitting (d=15)")
    fig_cmp.update_xaxes(title_text="Day of Year")
    fig_cmp.update_yaxes(title_text="Temperature (C)")
    st.plotly_chart(fig_cmp, use_container_width=True)

else:
    st.warning("Not enough data for selected city. Adjust sidebar filters.")

st.divider()

# ---------------------------------------------------------------------------
# 5. Code Example
# ---------------------------------------------------------------------------
code_example("""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Data: day_of_year -> temperature
X = df[['day_of_year']].values
y = df['temperature_c'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit polynomials of different degrees
for degree in [1, 3, 5, 10, 15]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)

    train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    print(f"Degree {degree:2d}: Train RMSE={train_rmse:.2f}, Test RMSE={test_rmse:.2f}")
""")

st.divider()

# ---------------------------------------------------------------------------
# 6. Quiz
# ---------------------------------------------------------------------------
quiz(
    "Your polynomial model has train RMSE = 0.5 but test RMSE = 8.2. What is happening?",
    [
        "Underfitting -- the model is too simple",
        "Overfitting -- the model memorized training data",
        "The data has no pattern",
        "The test set is corrupted",
    ],
    correct_idx=1,
    explanation="That massive gap between train and test error is the classic signature of overfitting. The model learned the training data so well that it memorized the noise along with the signal. When it encounters new data that has different noise, it fails spectacularly.",
    key="ch19_quiz1",
)

quiz(
    "Which polynomial degree would best fit temperature's annual cycle (sinusoidal)?",
    [
        "Degree 1 (linear)",
        "Degree 3-5 (low polynomial)",
        "Degree 15 (high polynomial)",
        "Degree 50 (very high polynomial)",
    ],
    correct_idx=1,
    explanation="A smooth seasonal curve only needs a few polynomial terms to approximate well. Degree 3-5 can capture the rise, peak, fall, and trough of an annual cycle without memorizing daily weather fluctuations. Higher degrees add wiggles that fit noise, not signal.",
    key="ch19_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 7. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Polynomial regression adds powers of X (x-squared, x-cubed, etc.) to capture non-linear relationships while still using the familiar linear regression machinery.",
    "The bias-variance tradeoff is the central tension: too simple = underfitting (model can't learn), too complex = overfitting (model memorizes noise). Finding the sweet spot is the art of modeling.",
    "ALWAYS evaluate on held-out test data. Training performance is a necessary but wildly insufficient measure of model quality. A model that scores 0.99 on training but 0.3 on test has learned nothing useful.",
    "For the annual temperature cycle, degree 3-5 is usually the sweet spot -- complex enough to capture seasonal patterns, simple enough to generalize to new data.",
    "The U-shaped test error curve is your guide: find the degree that minimizes test error, and you've found the right level of complexity for your data.",
])

navigation(
    prev_label="Ch 18: Multiple Regression",
    prev_page="18_Multiple_Regression.py",
    next_label="Ch 20: Regularization",
    next_page="20_Regularization.py",
)
