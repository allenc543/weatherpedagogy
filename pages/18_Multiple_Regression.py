"""Chapter 18: Multiple Regression — Multiple predictors, VIF, adjusted R-squared, interactions."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, color_map
from utils.ml_helpers import prepare_regression_data, regression_metrics
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)
from utils.constants import CITY_LIST, FEATURE_COLS, FEATURE_LABELS

# ---------------------------------------------------------------------------
st.set_page_config(page_title="Ch 18: Multiple Regression", layout="wide")
df = load_data()
fdf = sidebar_filters(df)

chapter_header(18, "Multiple Regression", part="IV")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "Multiple Regression: Because the World Has More Than One Cause",
    "Simple regression is nice, but it's also a bit naive. Temperature doesn't depend on just "
    "humidity -- it also depends on wind speed, atmospheric pressure, time of year, and time of "
    "day. Multiple regression lets you model all of these simultaneously. Instead of fitting a "
    "line through 2D space, you're fitting a <b>hyperplane</b> through high-dimensional space. "
    "The beautiful part: each coefficient now tells you the effect of that predictor <em>holding "
    "all others constant</em>. So when we say 'a 1% increase in humidity is associated with a "
    "0.15 C decrease in temperature,' we mean 'even after accounting for wind, pressure, and "
    "time of year.'",
)

col1, col2 = st.columns(2)
with col1:
    formula_box(
        "Multiple Regression Model",
        r"\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p",
        "Each beta_j represents the effect of x_j on y, holding all other predictors constant. This 'holding constant' part is the whole reason we bother with multiple regression instead of running separate simple regressions.",
    )
with col2:
    formula_box(
        "Adjusted R-squared",
        r"R^2_{\text{adj}} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}",
        "Here's the catch with regular R-squared: it NEVER decreases when you add a predictor, even a random noise column. Adjusted R-squared penalizes for the number of predictors, so it can actually go down if you add a useless variable. It's the honest version of R-squared.",
    )

st.markdown("### The Multicollinearity Trap")
concept_box(
    "Variance Inflation Factor (VIF): When Your Predictors Are Too Friendly",
    "Imagine you include both 'temperature in Fahrenheit' and 'temperature in Celsius' as "
    "predictors. They carry identical information, so the model can't figure out how to split "
    "the credit between them. Coefficients become wildly unstable -- change one data point "
    "and they swing dramatically. This is <b>multicollinearity</b>, and it's the silent killer "
    "of multiple regression. VIF quantifies how bad it is: a VIF of 1 means no collinearity, "
    "<b>VIF > 5</b> means 'you should investigate,' and <b>VIF > 10</b> means 'Houston, we "
    "have a problem.' Solutions: drop one of the correlated predictors, combine them, or use "
    "regularization (coming up in Chapter 20).",
)

formula_box(
    "VIF",
    r"\text{VIF}_j = \frac{1}{1 - R^2_j}",
    "where R-squared_j comes from regressing predictor x_j on all the OTHER predictors. If the other predictors can explain x_j well (high R-squared_j), then x_j is redundant, and VIF is high.",
)

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Build a Multiple Regression Model
# ---------------------------------------------------------------------------
st.subheader("Build It Yourself: Add and Remove Predictors")

st.markdown(
    "Let's predict **temperature** using various features. Check and uncheck predictors to "
    "see how R-squared and Adjusted R-squared change. The key question: does adding another "
    "predictor genuinely help, or are you just inflating R-squared with noise?"
)

target_var = "temperature_c"

available_predictors = ["relative_humidity_pct", "wind_speed_kmh", "surface_pressure_hpa", "month", "hour"]
predictor_labels = {
    **FEATURE_LABELS,
    "month": "Month (1-12)",
    "hour": "Hour (0-23)",
}

st.markdown("**Select predictors:**")
pred_cols = st.columns(len(available_predictors))
selected_preds = []
for i, pred in enumerate(available_predictors):
    with pred_cols[i]:
        if st.checkbox(predictor_labels.get(pred, pred), value=(i < 3), key=f"mr_pred_{pred}"):
            selected_preds.append(pred)

if len(selected_preds) == 0:
    st.warning("Select at least one predictor.")
else:
    # Prepare data
    model_data = fdf[selected_preds + [target_var]].dropna()

    if len(model_data) > 100:
        X = model_data[selected_preds].values
        y = model_data[target_var].values
        n, p = X.shape

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        r2 = model.score(X, y)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        residuals = y - y_pred
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))

        # Metrics
        met1, met2, met3, met4 = st.columns(4)
        met1.metric("R-squared", f"{r2:.4f}")
        met2.metric("Adjusted R-squared", f"{adj_r2:.4f}")
        met3.metric("RMSE", f"{rmse:.2f}")
        met4.metric("MAE", f"{mae:.2f}")

        # Coefficients table
        st.markdown("#### What Each Predictor Contributes")
        coef_data = []
        for j, pred in enumerate(selected_preds):
            coef_data.append({
                "Predictor": predictor_labels.get(pred, pred),
                "Coefficient": round(model.coef_[j], 4),
                "Interpretation": f"1 unit increase in {pred} -> {model.coef_[j]:.4f} change in temperature",
            })
        coef_data.append({
            "Predictor": "Intercept",
            "Coefficient": round(model.intercept_, 4),
            "Interpretation": "Baseline temperature when all predictors = 0",
        })
        coef_df = pd.DataFrame(coef_data)
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        # Coefficient bar chart
        fig_coef = go.Figure()
        labels = [predictor_labels.get(p, p) for p in selected_preds]
        colors = ["#2A9D8F" if c > 0 else "#E63946" for c in model.coef_]
        fig_coef.add_trace(go.Bar(
            x=labels, y=model.coef_,
            marker_color=colors,
        ))
        fig_coef.add_hline(y=0, line_dash="solid", line_color="gray")
        apply_common_layout(fig_coef, title="Regression Coefficients")
        fig_coef.update_yaxes(title_text="Coefficient Value")
        st.plotly_chart(fig_coef, use_container_width=True)

        # -------------------------------------------------------------------
        # VIF Calculation
        # -------------------------------------------------------------------
        st.markdown("#### Multicollinearity Check: VIF")

        if len(selected_preds) >= 2:
            vif_data = []
            for j, pred in enumerate(selected_preds):
                other_preds = [p for k, p in enumerate(selected_preds) if k != j]
                X_other = model_data[other_preds].values
                X_j = model_data[pred].values

                vif_model = LinearRegression()
                vif_model.fit(X_other, X_j)
                r2_j = vif_model.score(X_other, X_j)
                vif_val = 1 / (1 - r2_j) if r2_j < 1 else float("inf")

                vif_data.append({
                    "Predictor": predictor_labels.get(pred, pred),
                    "VIF": round(vif_val, 2),
                    "Status": "OK" if vif_val < 5 else "Caution" if vif_val < 10 else "Serious",
                })

            vif_df = pd.DataFrame(vif_data)
            st.dataframe(vif_df, use_container_width=True, hide_index=True)

            high_vif = [row for row in vif_data if row["VIF"] >= 5]
            if high_vif:
                warning_box(
                    f"{len(high_vif)} predictor(s) have VIF >= 5, which means they're substantially "
                    "correlated with the other predictors. The coefficients for these variables are "
                    "less trustworthy than they look -- small changes in the data could swing them "
                    "wildly. Consider removing one or using regularization (Chapter 20)."
                )
            else:
                insight_box("All VIF values are below 5. Your predictors are sufficiently independent of each other for the coefficients to be stable and interpretable.")
        else:
            st.info("VIF requires at least 2 predictors.")

        st.divider()

        # -------------------------------------------------------------------
        # R-squared Progression
        # -------------------------------------------------------------------
        st.subheader("Watch R-squared Grow (and Adjusted R-squared Keep It Honest)")

        st.markdown(
            "Here's the thing about R-squared that makes statisticians nervous: it can ONLY go up "
            "when you add predictors. Even a column of pure random noise will increase R-squared "
            "(slightly). Adjusted R-squared is the fix -- it penalizes for each additional predictor, "
            "so it can actually decrease if a new variable adds more complexity than explanatory power."
        )

        progression = []
        for k in range(1, len(selected_preds) + 1):
            sub_preds = selected_preds[:k]
            X_sub = model_data[sub_preds].values
            m_sub = LinearRegression()
            m_sub.fit(X_sub, y)
            r2_sub = m_sub.score(X_sub, y)
            adj_r2_sub = 1 - (1 - r2_sub) * (n - 1) / (n - k - 1)
            progression.append({
                "Predictors": " + ".join([predictor_labels.get(p, p) for p in sub_preds]),
                "Num Predictors": k,
                "R-squared": round(r2_sub, 4),
                "Adj R-squared": round(adj_r2_sub, 4),
                "R-squared Gain": round(r2_sub - (progression[-1]["R-squared"] if progression else 0), 4) if k > 1 else round(r2_sub, 4),
            })

        prog_df = pd.DataFrame(progression)
        st.dataframe(prog_df, use_container_width=True, hide_index=True)

        fig_prog = go.Figure()
        fig_prog.add_trace(go.Scatter(
            x=prog_df["Num Predictors"], y=prog_df["R-squared"],
            mode="lines+markers", name="R-squared",
            line=dict(color="#2A9D8F", width=2), marker=dict(size=8),
        ))
        fig_prog.add_trace(go.Scatter(
            x=prog_df["Num Predictors"], y=prog_df["Adj R-squared"],
            mode="lines+markers", name="Adjusted R-squared",
            line=dict(color="#E63946", width=2, dash="dash"), marker=dict(size=8),
        ))
        apply_common_layout(fig_prog, title="R-squared vs Number of Predictors")
        fig_prog.update_xaxes(title_text="Number of Predictors", dtick=1)
        fig_prog.update_yaxes(title_text="R-squared", range=[0, 1])
        st.plotly_chart(fig_prog, use_container_width=True)

        insight_box(
            "See the gap between the two lines? R-squared always goes up, but adjusted R-squared "
            "sometimes flattens or even dips. When adjusted R-squared stops increasing meaningfully, "
            "that's your signal: additional predictors are adding complexity without real explanatory power. "
            "This is the principle of parsimony -- prefer the simplest model that does the job."
        )

        # -------------------------------------------------------------------
        # Actual vs Predicted
        # -------------------------------------------------------------------
        st.divider()
        st.subheader("The Reality Check: Actual vs Predicted")

        if len(y) > 5000:
            plot_idx = np.random.RandomState(42).choice(len(y), 5000, replace=False)
        else:
            plot_idx = np.arange(len(y))

        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=y[plot_idx], y=y_pred[plot_idx],
            mode="markers", marker=dict(color="#2A9D8F", size=3, opacity=0.2),
            showlegend=False,
        ))
        min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        fig_avp.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", line=dict(color="#E63946", dash="dash"),
            name="Perfect Fit",
        ))
        apply_common_layout(fig_avp, title="Actual vs Predicted Temperature")
        fig_avp.update_xaxes(title_text="Actual Temperature (C)")
        fig_avp.update_yaxes(title_text="Predicted Temperature (C)")
        st.plotly_chart(fig_avp, use_container_width=True)

    else:
        st.warning("Not enough data. Adjust sidebar filters.")

st.divider()

# ---------------------------------------------------------------------------
# 3. Code Example
# ---------------------------------------------------------------------------
code_example("""
from sklearn.linear_model import LinearRegression
import numpy as np

# Multiple regression: predict temperature
features = ['relative_humidity_pct', 'wind_speed_kmh', 'surface_pressure_hpa', 'month']
X = df[features].values
y = df['temperature_c'].values

model = LinearRegression()
model.fit(X, y)

# Results
print(f"R-squared: {model.score(X, y):.4f}")
n, p = X.shape
adj_r2 = 1 - (1 - model.score(X, y)) * (n-1) / (n-p-1)
print(f"Adjusted R-squared: {adj_r2:.4f}")

# Coefficients
for feat, coef in zip(features, model.coef_):
    print(f"  {feat}: {coef:.4f}")

# VIF calculation
from sklearn.linear_model import LinearRegression as LR
for j, feat in enumerate(features):
    others = [f for i, f in enumerate(features) if i != j]
    r2_j = LR().fit(df[others], df[feat]).score(df[others], df[feat])
    vif = 1 / (1 - r2_j)
    print(f"VIF({feat}) = {vif:.2f}")
""")

st.divider()

# ---------------------------------------------------------------------------
# 4. Quiz
# ---------------------------------------------------------------------------
quiz(
    "You add a random noise column to your regression. What happens?",
    [
        "R-squared decreases",
        "R-squared stays exactly the same",
        "R-squared increases slightly, but Adjusted R-squared may decrease",
        "Both R-squared and Adjusted R-squared increase",
    ],
    correct_idx=2,
    explanation="R-squared can never decrease when you add a predictor -- even random noise will fit to some tiny amount of variability by chance. But adjusted R-squared knows better: it penalizes for the added complexity, and since the noise column doesn't actually help, adjusted R-squared may drop. This is exactly why adjusted R-squared exists.",
    key="ch18_quiz1",
)

quiz(
    "A predictor has VIF = 12. This means:",
    [
        "It has a strong effect on the target",
        "It is highly correlated with other predictors, making its coefficient unstable",
        "It should definitely be removed",
        "The model is overfit",
    ],
    correct_idx=1,
    explanation="VIF = 12 means this predictor's variance is inflated 12x compared to what it would be if it were uncorrelated with the others. The coefficient estimate is unreliable -- but that doesn't necessarily mean you should remove it. Maybe you can combine it with the correlated predictor, or use regularization to stabilize things.",
    key="ch18_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 5. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Multiple regression models Y as a function of several predictors simultaneously. Each coefficient represents 'the effect of this variable, holding everything else constant.'",
    "Adjusted R-squared is the honest version of R-squared: it penalizes for unnecessary predictors. When adjusted R-squared starts declining, you've added too many variables.",
    "Multicollinearity (correlated predictors) makes coefficients unstable and uninterpretable. VIF quantifies the problem: > 5 is concerning, > 10 is serious.",
    "R-squared always increases when you add predictors, even useless ones. This is why you need adjusted R-squared -- it can tell you when more complexity isn't helping.",
    "The order in which you add predictors matters for the R-squared progression. Variables that explain the same variation as previously-added ones will show diminishing marginal returns.",
])

navigation(
    prev_label="Ch 17: Simple Linear Regression",
    prev_page="17_Simple_Linear_Regression.py",
    next_label="Ch 19: Polynomial Regression",
    next_page="19_Polynomial_Regression.py",
)
