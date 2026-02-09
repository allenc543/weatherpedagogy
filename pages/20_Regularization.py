"""Chapter 20: Regularization (Ridge & Lasso) — L1/L2 penalties, coefficient shrinkage, feature selection."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, color_map
from utils.ml_helpers import regression_metrics
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)
from utils.constants import CITY_LIST, FEATURE_COLS, FEATURE_LABELS

# ---------------------------------------------------------------------------
st.set_page_config(page_title="Ch 20: Regularization", layout="wide")
df = load_data()
fdf = sidebar_filters(df)

chapter_header(20, "Regularization (Ridge & Lasso)", part="IV")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "Regularization: Teaching Your Model Some Self-Restraint",
    "Here's the problem with unregularized regression: when you have lots of features or "
    "high-degree polynomials, the model gets greedy. It assigns enormous coefficients to fit "
    "every little wiggle in the training data, which means it's memorizing noise instead of "
    "learning patterns. Regularization is essentially telling the model: 'You can use these "
    "features, but there's a tax on large coefficients.' The model then has to balance two "
    "competing goals -- fitting the data well AND keeping coefficients small. This trades a "
    "small amount of <b>bias</b> (the model can't fit the training data quite as well) for a "
    "large reduction in <b>variance</b> (the model generalizes much better to new data).",
)

col1, col2 = st.columns(2)
with col1:
    formula_box(
        "Ridge Regression (L2 penalty)",
        r"\min_\beta \sum (y_i - X_i \beta)^2 + \alpha \sum_{j=1}^{p} \beta_j^2",
        "Ridge adds a tax proportional to the SQUARE of each coefficient. This shrinks everything toward zero, but never all the way. It's the polite form of regularization -- it tells each coefficient to calm down, but doesn't kick any of them out entirely.",
    )
with col2:
    formula_box(
        "Lasso Regression (L1 penalty)",
        r"\min_\beta \sum (y_i - X_i \beta)^2 + \alpha \sum_{j=1}^{p} |\beta_j|",
        "Lasso uses the ABSOLUTE value instead of the square. This seemingly minor change has a dramatic consequence: Lasso can drive coefficients all the way to exactly zero, effectively removing features from the model. It's regularization that does feature selection for free.",
    )

st.markdown("### Ridge vs Lasso: A Comparison of Philosophies")
comp_df = pd.DataFrame({
    "Property": ["Penalty", "Coefficient shrinkage", "Feature selection", "Correlated features", "When to use"],
    "Ridge (L2)": [
        "Sum of squared coefficients",
        "Shrinks toward zero, never exactly zero",
        "No (keeps all features, just quieter)",
        "Shares weight among correlated features",
        "When you believe many features have small/medium effects",
    ],
    "Lasso (L1)": [
        "Sum of absolute coefficients",
        "Can shrink all the way to exactly zero",
        "Yes -- automatic and built-in",
        "Picks one correlated feature, zeros out the rest",
        "When you suspect only a few features actually matter",
    ],
})
st.dataframe(comp_df, use_container_width=True, hide_index=True)

formula_box(
    "The alpha hyperparameter: How Much Restraint?",
    r"\alpha = 0 \implies \text{OLS (no penalty)} \qquad \alpha \to \infty \implies \text{all } \beta_j \to 0",
    "Alpha is the dial that controls how much you penalize large coefficients. Alpha = 0 means no regularization (you're back to OLS). Alpha = infinity means all coefficients get squashed to zero (you're predicting the mean for everyone). Somewhere in between is the sweet spot.",
)

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Coefficient Paths (alpha sweep)
# ---------------------------------------------------------------------------
st.subheader("Watch the Coefficients Shrink: The Regularization Path")

st.markdown(
    "We're predicting **temperature** from humidity, wind speed, pressure, month, and hour. "
    "As alpha increases (moving right), the penalty gets stronger and the coefficients shrink. "
    "With Ridge, they shrink smoothly toward zero. With Lasso, they hit zero and stay there -- "
    "that's the feature selection happening in real time."
)

features = ["relative_humidity_pct", "wind_speed_kmh", "surface_pressure_hpa", "month", "hour"]
feature_display = {
    **FEATURE_LABELS,
    "month": "Month",
    "hour": "Hour",
}

reg_data = fdf[features + ["temperature_c"]].dropna()

if len(reg_data) > 100:
    X_raw = reg_data[features].values
    y_raw = reg_data["temperature_c"].values

    # Standardize so coefficients are comparable
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_raw, test_size=0.2, random_state=42
    )

    # Alpha range (log scale)
    alphas = np.logspace(-3, 4, 80)

    ridge_coefs = []
    lasso_coefs = []
    ridge_test_r2 = []
    lasso_test_r2 = []

    for a in alphas:
        # Ridge
        ridge = Ridge(alpha=a)
        ridge.fit(X_train, y_train)
        ridge_coefs.append(ridge.coef_.copy())
        ridge_test_r2.append(ridge.score(X_test, y_test))

        # Lasso
        lasso = Lasso(alpha=a, max_iter=10000)
        lasso.fit(X_train, y_train)
        lasso_coefs.append(lasso.coef_.copy())
        lasso_test_r2.append(lasso.score(X_test, y_test))

    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)

    # Plot coefficient paths
    path_col1, path_col2 = st.columns(2)

    colors_list = ["#E63946", "#2A9D8F", "#264653", "#F4A261", "#7209B7"]

    with path_col1:
        fig_ridge = go.Figure()
        for j, feat in enumerate(features):
            fig_ridge.add_trace(go.Scatter(
                x=alphas, y=ridge_coefs[:, j],
                mode="lines", name=feature_display.get(feat, feat),
                line=dict(color=colors_list[j % len(colors_list)], width=2),
            ))
        fig_ridge.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
        apply_common_layout(fig_ridge, title="Ridge: Coefficient Paths")
        fig_ridge.update_xaxes(title_text="Alpha (log scale)", type="log")
        fig_ridge.update_yaxes(title_text="Coefficient Value")
        st.plotly_chart(fig_ridge, use_container_width=True)

    with path_col2:
        fig_lasso = go.Figure()
        for j, feat in enumerate(features):
            fig_lasso.add_trace(go.Scatter(
                x=alphas, y=lasso_coefs[:, j],
                mode="lines", name=feature_display.get(feat, feat),
                line=dict(color=colors_list[j % len(colors_list)], width=2),
            ))
        fig_lasso.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
        apply_common_layout(fig_lasso, title="Lasso: Coefficient Paths")
        fig_lasso.update_xaxes(title_text="Alpha (log scale)", type="log")
        fig_lasso.update_yaxes(title_text="Coefficient Value")
        st.plotly_chart(fig_lasso, use_container_width=True)

    insight_box(
        "Look at the difference. Ridge coefficients approach zero asymptotically -- they get "
        "closer and closer but never quite arrive. Lasso coefficients hit zero at different "
        "alpha values and flatline. The order in which features get zeroed out tells you "
        "something: the first to go are the least important. It's like a reality show where "
        "contestants get eliminated one by one, except the elimination criterion is 'contribution "
        "to prediction accuracy.'"
    )

    st.divider()

    # -------------------------------------------------------------------
    # 3. Interactive: Single Alpha Control
    # -------------------------------------------------------------------
    st.subheader("Pick an Alpha and See What Happens")

    reg_type = st.radio("Regularization type", ["Ridge (L2)", "Lasso (L1)"], key="reg_type")

    alpha_val = st.slider(
        "Alpha (regularization strength)",
        min_value=-3.0, max_value=4.0, value=0.0, step=0.1,
        format="10^%.1f",
        key="alpha_slider",
        help="Log-scale: 10^(-3) to 10^4",
    )
    alpha_actual = 10 ** alpha_val

    st.markdown(f"**Alpha = {alpha_actual:.4f}**")

    if "Ridge" in reg_type:
        model = Ridge(alpha=alpha_actual)
    else:
        model = Lasso(alpha=alpha_actual, max_iter=10000)

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_m = regression_metrics(y_train, y_train_pred)
    test_m = regression_metrics(y_test, y_test_pred)

    # Also compute OLS baseline
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    ols_test_r2 = ols.score(X_test, y_test)

    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Train R-squared", f"{train_m['r2']:.4f}")
    met2.metric("Test R-squared", f"{test_m['r2']:.4f}")
    met3.metric("Test RMSE", f"{test_m['rmse']:.2f}")
    met4.metric("OLS Test R-squared", f"{ols_test_r2:.4f}")

    # Coefficient comparison table
    coef_table = pd.DataFrame({
        "Feature": [feature_display.get(f, f) for f in features],
        "OLS Coefficient": [round(c, 4) for c in ols.coef_],
        f"{reg_type.split(' ')[0]} Coefficient": [round(c, 4) for c in model.coef_],
        "Shrinkage (%)": [
            round((1 - abs(model.coef_[j]) / max(abs(ols.coef_[j]), 1e-10)) * 100, 1)
            for j in range(len(features))
        ],
        "Zeroed Out": [abs(c) < 1e-10 for c in model.coef_],
    })
    st.dataframe(coef_table, use_container_width=True, hide_index=True)

    n_zeroed = sum(abs(c) < 1e-10 for c in model.coef_)
    if n_zeroed > 0:
        insight_box(
            f"Lasso has driven {n_zeroed} feature(s) to exactly zero at this alpha level. "
            "These features have been judged 'not worth their complexity cost' by the algorithm. "
            "This is automatic feature selection -- no human had to decide which variables to drop."
        )

    # Coefficient bar chart: OLS vs Regularized
    fig_bar = go.Figure()
    feat_labels = [feature_display.get(f, f) for f in features]
    fig_bar.add_trace(go.Bar(
        x=feat_labels, y=ols.coef_, name="OLS", marker_color="#264653", opacity=0.7,
    ))
    fig_bar.add_trace(go.Bar(
        x=feat_labels, y=model.coef_, name=reg_type.split(" ")[0],
        marker_color="#E63946", opacity=0.7,
    ))
    fig_bar.add_hline(y=0, line_dash="solid", line_color="gray")
    apply_common_layout(fig_bar, title=f"OLS vs {reg_type.split(' ')[0]} Coefficients (alpha={alpha_actual:.4f})")
    fig_bar.update_yaxes(title_text="Coefficient (standardized)")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # -------------------------------------------------------------------
    # 4. Test R-squared vs Alpha
    # -------------------------------------------------------------------
    st.subheader("Finding the Sweet Spot: Test Performance vs Alpha")

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=alphas, y=ridge_test_r2, mode="lines", name="Ridge",
        line=dict(color="#2A9D8F", width=2),
    ))
    fig_perf.add_trace(go.Scatter(
        x=alphas, y=lasso_test_r2, mode="lines", name="Lasso",
        line=dict(color="#E63946", width=2),
    ))
    fig_perf.add_hline(y=ols_test_r2, line_dash="dash", line_color="#264653",
                       annotation_text=f"OLS R-squared = {ols_test_r2:.4f}")
    fig_perf.add_vline(x=alpha_actual, line_dash="dot", line_color="#F4A261",
                       annotation_text=f"Your alpha = {alpha_actual:.4f}")
    apply_common_layout(fig_perf, title="Test R-squared vs Regularization Strength")
    fig_perf.update_xaxes(title_text="Alpha (log scale)", type="log")
    fig_perf.update_yaxes(title_text="Test R-squared")
    st.plotly_chart(fig_perf, use_container_width=True)

    best_ridge_alpha = alphas[np.argmax(ridge_test_r2)]
    best_lasso_alpha = alphas[np.argmax(lasso_test_r2)]
    st.markdown(
        f"**Best Ridge alpha:** {best_ridge_alpha:.4f} (R-squared = {max(ridge_test_r2):.4f})  \n"
        f"**Best Lasso alpha:** {best_lasso_alpha:.4f} (R-squared = {max(lasso_test_r2):.4f})"
    )

    st.divider()

    # -------------------------------------------------------------------
    # 5. Polynomial + Regularization Demo
    # -------------------------------------------------------------------
    st.subheader("The Punchline: Regularization Tames Overfitting")

    st.markdown(
        "Remember the overfitting problem from Chapter 19? High-degree polynomials go wild, "
        "oscillating between data points. Here's the elegant fix: slap a Ridge penalty on those "
        "coefficients. The polynomial still has the *flexibility* to fit complex curves, but "
        "the penalty prevents the coefficients from getting absurdly large, which is what causes "
        "the wild oscillations."
    )

    bonus_data = fdf[fdf["city"] == CITY_LIST[0]][["day_of_year", "temperature_c"]].dropna()

    if len(bonus_data) > 100:
        X_b = bonus_data["day_of_year"].values.reshape(-1, 1)
        y_b = bonus_data["temperature_c"].values
        X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_b, y_b, test_size=0.2, random_state=42)

        deg_bonus = 12
        poly_feat = PolynomialFeatures(deg_bonus, include_bias=False)
        X_b_train_poly = poly_feat.fit_transform(X_b_train)
        X_b_test_poly = poly_feat.transform(X_b_test)

        # Standardize polynomial features
        sc_b = StandardScaler()
        X_b_train_sc = sc_b.fit_transform(X_b_train_poly)
        X_b_test_sc = sc_b.transform(X_b_test_poly)

        x_curve_b = np.linspace(1, 366, 500).reshape(-1, 1)
        x_curve_poly = poly_feat.transform(x_curve_b)
        x_curve_sc = sc_b.transform(x_curve_poly)

        # OLS
        ols_b = LinearRegression().fit(X_b_train_sc, y_b_train)
        ols_curve = ols_b.predict(x_curve_sc)
        ols_r2 = ols_b.score(X_b_test_sc, y_b_test)

        # Ridge
        ridge_b = Ridge(alpha=100).fit(X_b_train_sc, y_b_train)
        ridge_curve = ridge_b.predict(x_curve_sc)
        ridge_r2 = ridge_b.score(X_b_test_sc, y_b_test)

        if len(X_b_test) > 1000:
            pidx = np.random.RandomState(42).choice(len(X_b_test), 1000, replace=False)
        else:
            pidx = np.arange(len(X_b_test))

        fig_bonus = go.Figure()
        fig_bonus.add_trace(go.Scatter(
            x=X_b_test[pidx].flatten(), y=y_b_test[pidx],
            mode="markers", name="Test Data",
            marker=dict(color="gray", size=3, opacity=0.3),
        ))
        fig_bonus.add_trace(go.Scatter(
            x=x_curve_b.flatten(), y=ols_curve,
            mode="lines", name=f"OLS Degree {deg_bonus} (test R2={ols_r2:.3f})",
            line=dict(color="#E63946", width=2, dash="dash"),
        ))
        fig_bonus.add_trace(go.Scatter(
            x=x_curve_b.flatten(), y=ridge_curve,
            mode="lines", name=f"Ridge Degree {deg_bonus} (test R2={ridge_r2:.3f})",
            line=dict(color="#2A9D8F", width=2),
        ))
        apply_common_layout(fig_bonus, title=f"Degree {deg_bonus}: OLS (Overfit) vs Ridge (Regularized)")
        fig_bonus.update_xaxes(title_text="Day of Year")
        fig_bonus.update_yaxes(title_text="Temperature (C)")
        st.plotly_chart(fig_bonus, use_container_width=True)

        insight_box(
            f"Same degree-{deg_bonus} polynomial, dramatically different behavior. OLS goes haywire "
            f"(test R-squared = {ols_r2:.3f}) because nothing stops the coefficients from exploding. "
            f"Ridge keeps them in check (test R-squared = {ridge_r2:.3f}), producing a smooth curve "
            f"that actually captures the seasonal pattern. This is the power of regularization: you "
            f"get the flexibility of a complex model with the stability of a simple one."
        )

else:
    st.warning("Not enough data. Adjust sidebar filters.")

st.divider()

# ---------------------------------------------------------------------------
# 6. Code Example
# ---------------------------------------------------------------------------
code_example("""
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Prepare data (always standardize for regularization!)
features = ['relative_humidity_pct', 'wind_speed_kmh', 'surface_pressure_hpa', 'month', 'hour']
X = df[features].values
y = df['temperature_c'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"Ridge R2: {ridge.score(X_test, y_test):.4f}")
print(f"Ridge coefs: {ridge.coef_}")

# Lasso (performs feature selection)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"Lasso R2: {lasso.score(X_test, y_test):.4f}")
print(f"Lasso coefs: {lasso.coef_}")
print(f"Features zeroed out: {sum(abs(c) < 1e-10 for c in lasso.coef_)}")

# Coefficient path
alphas = np.logspace(-3, 3, 50)
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000).fit(X_train, y_train)
    n_nonzero = sum(abs(c) > 1e-10 for c in lasso.coef_)
    print(f"alpha={alpha:.4f}: {n_nonzero} features, R2={lasso.score(X_test, y_test):.4f}")
""")

st.divider()

# ---------------------------------------------------------------------------
# 7. Quiz
# ---------------------------------------------------------------------------
quiz(
    "Which regularization method can set coefficients exactly to zero?",
    [
        "Ridge (L2)",
        "Lasso (L1)",
        "Both Ridge and Lasso",
        "Neither",
    ],
    correct_idx=1,
    explanation="Lasso uses the absolute value penalty, which has a geometric property that causes some coefficients to land exactly at zero. Ridge uses the squared penalty, which shrinks coefficients toward zero asymptotically but never gets there. This is the key practical difference: Lasso does feature selection, Ridge doesn't.",
    key="ch20_quiz1",
)

quiz(
    "What happens as alpha approaches infinity in Ridge regression?",
    [
        "All coefficients become very large",
        "The model becomes equivalent to OLS",
        "All coefficients approach zero (the model predicts the mean)",
        "The model automatically selects the best features",
    ],
    correct_idx=2,
    explanation="As the penalty becomes infinitely strong, any nonzero coefficient incurs an infinite cost. The only way to minimize the total loss is to set all coefficients to zero, which leaves only the intercept -- and the intercept is just the mean of y. So infinite Ridge penalty turns your sophisticated regression model into a simple mean predictor. Not very useful, but it proves a point about the bias-variance tradeoff.",
    key="ch20_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 8. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Regularization adds a 'complexity tax' to prevent overfitting. It deliberately introduces a tiny bit of bias in exchange for a large reduction in variance -- one of the best trades in all of statistics.",
    "Ridge (L2) shrinks all coefficients toward zero but never reaches it. Lasso (L1) can zero out coefficients entirely, performing automatic feature selection. The choice between them depends on whether you think many features have small effects (Ridge) or a few features have large effects (Lasso).",
    "Alpha is the dial that controls the penalty strength. Alpha = 0 means OLS (no regularization). Alpha = infinity means 'predict the mean for everyone.' The optimal alpha is somewhere in between, and cross-validation is how you find it.",
    "ALWAYS standardize your features before applying regularization. If features are on different scales, the penalty will unfairly punish the ones with larger values.",
    "The killer app of regularization: it lets you use complex models (high-degree polynomials, many features) without the overfitting that would normally destroy their performance. Flexibility with discipline.",
])

navigation(
    prev_label="Ch 19: Polynomial Regression",
    prev_page="19_Polynomial_Regression.py",
    next_label="Ch 21: Logistic Regression",
    next_page="21_Logistic_Regression.py",
)
