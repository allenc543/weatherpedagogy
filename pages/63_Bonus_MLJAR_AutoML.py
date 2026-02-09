"""Bonus Chapter: MLJAR AutoML -- Let a machine build the machine learning pipeline."""
import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.ml_helpers import classification_metrics, regression_metrics, plot_confusion_matrix
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(63, "Bonus: MLJAR AutoML", part="Bonus")

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: What is AutoML and why should you care
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "Over the past 62 chapters, you've done something remarkable: you've manually chosen "
    "algorithms, tuned hyperparameters, engineered features, compared models, and built "
    "ensembles. You've learned *why* each decision matters. But let me ask you a slightly "
    "uncomfortable question: what if a computer could do all of that automatically?"
)
st.markdown(
    "That's what **AutoML** (Automated Machine Learning) does. You give it a dataset and "
    "a target column. It tries dozens of algorithms, tunes their hyperparameters, engineers "
    "features, builds ensembles, and hands you back the best model it found. The whole "
    "pipeline you've been learning to build by hand -- automated."
)
st.markdown(
    "**MLJAR** (specifically `mljar-supervised`) is one of the best open-source AutoML "
    "libraries. It's opinionated in a good way: it tries a curated set of algorithms "
    "(linear models, random forests, XGBoost, LightGBM, CatBoost, neural networks), "
    "performs feature engineering, builds stacking ensembles, and produces a human-readable "
    "leaderboard. It's what a senior data scientist would do if they had infinite patience."
)

concept_box(
    "Why Learn ML If AutoML Exists?",
    "Fair question. Here's the honest answer: AutoML is a power tool, not a replacement "
    "for understanding. A power drill doesn't replace knowing what a screw is. You need "
    "to understand the concepts we've covered to:<br>"
    "- <b>Know when AutoML is appropriate</b> (tabular data: yes. Real-time streaming: maybe not.)<br>"
    "- <b>Interpret the results</b> (why did it choose XGBoost over logistic regression? "
    "Is the feature importance sensible?)<br>"
    "- <b>Spot problems</b> (data leakage, target leakage, overfitting to a small dataset)<br>"
    "- <b>Debug failures</b> (AutoML isn't magic. It can produce garbage if the data has issues.)<br><br>"
    "Think of AutoML as the final boss that validates everything you've learned. If you "
    "understand chapters 1-62, you can use AutoML as a teammate. If you don't, you're "
    "just running code you don't understand.",
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Task 1 — City Classification
# ─────────────────────────────────────────────────────────────────────────────
st.header("Task 1: Predict Which City (Classification)")

st.markdown(
    "Let's start with our old friend: the city classification task. Here's the exact setup, "
    "same as every classification chapter:"
)
st.markdown(
    "**Input**: 4 numbers from a single hourly weather reading:\n"
    "- `temperature_c`: temperature in Celsius (e.g., 28.3°C)\n"
    "- `relative_humidity_pct`: humidity as a percentage (e.g., 72%)\n"
    "- `wind_speed_kmh`: wind speed in km/h (e.g., 8.4)\n"
    "- `surface_pressure_hpa`: atmospheric pressure in hectopascals (e.g., 1015.2)\n\n"
    "**Output**: Which of the 6 cities (Dallas, San Antonio, Houston, Austin, NYC, LA) "
    "this reading came from.\n\n"
    "**The question**: How does MLJAR's fully automated approach compare to the models "
    "we hand-built in chapters 21-49?"
)

# Prepare classification data
clf_data = fdf[FEATURE_COLS + ["city"]].dropna()

# Subsample for reasonable runtime (MLJAR can be slow on 100K+ rows)
clf_sample_size = st.slider(
    "Training sample size (larger = better models but slower)",
    min_value=2000, max_value=20000, value=5000, step=1000,
    key="clf_sample",
    help="MLJAR tries many models. 5,000 rows takes ~1-3 minutes. 20,000 takes longer."
)

clf_sample = clf_data.sample(min(clf_sample_size, len(clf_data)), random_state=42)
X_clf = clf_sample[FEATURE_COLS]
y_clf = clf_sample["city"]

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

st.markdown(
    f"**Dataset**: {len(X_clf_train)} training rows, {len(X_clf_test)} test rows, "
    f"4 features, 6 city classes."
)

# MLJAR mode selection
clf_mode = st.selectbox(
    "MLJAR mode",
    ["Explain", "Perform", "Compete"],
    key="clf_mode",
    help="Explain = fast, good for understanding. Perform = balanced. Compete = maximum accuracy, slowest."
)

st.markdown(
    f"**Mode: {clf_mode}**\n"
    f"- **Explain**: Tries a few algorithms with default settings. Fast (~30 sec). "
    f"Good for a quick look.\n"
    f"- **Perform**: More algorithms, more tuning, ensembles. A few minutes.\n"
    f"- **Compete**: Everything including stacking ensembles and hill climbing. "
    f"The kitchen sink. Can take 5-10+ minutes."
)

# Output directory for MLJAR
clf_output_dir = "/tmp/mljar_clf_weather"

run_clf = st.button("Run MLJAR on City Classification", key="run_clf", type="primary")

if run_clf:
    # Clean previous run
    if os.path.exists(clf_output_dir):
        shutil.rmtree(clf_output_dir)

    with st.spinner(f"MLJAR is trying dozens of models in '{clf_mode}' mode. This may take a minute..."):
        try:
            from supervised import AutoML

            automl_clf = AutoML(
                mode=clf_mode,
                results_path=clf_output_dir,
                total_time_limit=300,  # 5 min safety cap
                random_state=42,
                ml_task="multiclass_classification",
            )
            automl_clf.fit(X_clf_train, y_clf_train)

            # Store in session state so results persist
            st.session_state["automl_clf"] = automl_clf
            st.session_state["clf_test_data"] = (X_clf_test, y_clf_test)
            st.session_state["clf_output_dir"] = clf_output_dir

        except ImportError:
            st.error(
                "**mljar-supervised is not installed.** Run: `pip install mljar-supervised` "
                "and restart the app."
            )
        except Exception as e:
            st.error(f"MLJAR encountered an error: {e}")

# Show results if we have them
if "automl_clf" in st.session_state:
    automl_clf = st.session_state["automl_clf"]
    X_clf_test, y_clf_test = st.session_state["clf_test_data"]

    st.subheader("MLJAR Leaderboard")
    st.markdown(
        "This is every model MLJAR tried, ranked by validation score. MLJAR uses "
        "internal cross-validation, so these scores are honest (no peeking at test data)."
    )

    # Get leaderboard
    try:
        leaderboard = automl_clf.get_leaderboard()
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)
    except Exception:
        st.info("Leaderboard display not available in this MLJAR version.")

    # Test set predictions
    y_clf_pred = automl_clf.predict(X_clf_test)
    clf_test_acc = accuracy_score(y_clf_test, y_clf_pred)

    st.subheader("Test Set Performance (Data MLJAR Has Never Seen)")

    col1, col2, col3 = st.columns(3)
    col1.metric("MLJAR Test Accuracy", f"{clf_test_acc:.4f}")
    col2.metric("Random Guessing Baseline", "0.1667", help="1/6 for 6 classes")
    col3.metric("Improvement Over Random", f"{clf_test_acc - 1/6:.4f}")

    # Confusion matrix
    city_labels = sorted(y_clf_test.unique())
    cm = confusion_matrix(y_clf_test, y_clf_pred, labels=city_labels)
    fig_cm = plot_confusion_matrix(cm, city_labels)
    fig_cm.update_layout(title="MLJAR City Classification — Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown(
        "Look at the confusion matrix. The same city pairs that confused our hand-built "
        "models in earlier chapters will likely confuse MLJAR too. Dallas ↔ San Antonio, "
        "Houston ↔ Austin -- these are fundamentally hard to distinguish from 4 weather "
        "features because the cities genuinely have similar climates. AutoML can't overcome "
        "the information limits of the data itself."
    )

    # Per-city accuracy
    per_city_acc = {}
    for city in city_labels:
        mask = y_clf_test == city
        if mask.sum() > 0:
            per_city_acc[city] = accuracy_score(y_clf_test[mask], y_clf_pred[mask])

    city_acc_df = pd.DataFrame({
        "City": list(per_city_acc.keys()),
        "Accuracy": list(per_city_acc.values()),
    }).sort_values("Accuracy", ascending=False)

    fig_city = px.bar(city_acc_df, x="City", y="Accuracy",
                      color="City", color_discrete_map=CITY_COLORS,
                      title="MLJAR Classification Accuracy by City")
    apply_common_layout(fig_city, height=400)
    st.plotly_chart(fig_city, use_container_width=True)

    insight_box(
        f"MLJAR achieved {clf_test_acc:.1%} test accuracy on the city classification task. "
        f"Compare this to the models you built by hand in chapters 21-49. If MLJAR is "
        f"similar or slightly better, that's normal — it has the advantage of trying more "
        f"algorithms and tuning more carefully. If it's dramatically better, the gap is "
        f"probably from ensemble stacking (Ch 49) or feature engineering you didn't try. "
        f"If your hand-built model is close, congratulations — you've learned the craft well."
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Task 2 — Temperature Regression
# ─────────────────────────────────────────────────────────────────────────────
st.header("Task 2: Predict Temperature (Regression)")

st.markdown(
    "Now let's try a regression task. Instead of predicting which city, we'll predict "
    "the temperature from the other 3 weather features plus time features."
)
st.markdown(
    "**Input**: humidity, wind speed, pressure, plus the hour of day and month "
    "(so the model knows about diurnal and seasonal cycles).\n\n"
    "**Output**: temperature in Celsius.\n\n"
    "**Why this is interesting**: In the regression chapters (17-20), we built polynomial "
    "models and multi-feature linear regressions. MLJAR will try all of that plus XGBoost, "
    "LightGBM, neural networks, and ensembles. Can automation beat our hand-tuned models?"
)

reg_city = st.selectbox("City for regression", CITY_LIST, key="reg_city")
reg_data = fdf[fdf["city"] == reg_city][
    ["relative_humidity_pct", "wind_speed_kmh", "surface_pressure_hpa", "hour", "month", "temperature_c"]
].dropna()

reg_sample_size = st.slider(
    "Training sample size for regression",
    min_value=2000, max_value=15000, value=5000, step=1000,
    key="reg_sample",
)

reg_sample = reg_data.sample(min(reg_sample_size, len(reg_data)), random_state=42)
reg_features = ["relative_humidity_pct", "wind_speed_kmh", "surface_pressure_hpa", "hour", "month"]
X_reg = reg_sample[reg_features]
y_reg = reg_sample["temperature_c"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

st.markdown(
    f"**Dataset**: {len(X_reg_train)} training rows, {len(X_reg_test)} test rows for {reg_city}. "
    f"Features: humidity, wind speed, pressure, hour, month → predict temperature."
)

reg_mode = st.selectbox(
    "MLJAR mode for regression",
    ["Explain", "Perform", "Compete"],
    key="reg_mode",
)

reg_output_dir = "/tmp/mljar_reg_weather"
run_reg = st.button("Run MLJAR on Temperature Prediction", key="run_reg", type="primary")

if run_reg:
    if os.path.exists(reg_output_dir):
        shutil.rmtree(reg_output_dir)

    with st.spinner(f"MLJAR is building regression models in '{reg_mode}' mode..."):
        try:
            from supervised import AutoML

            automl_reg = AutoML(
                mode=reg_mode,
                results_path=reg_output_dir,
                total_time_limit=300,
                random_state=42,
                ml_task="regression",
            )
            automl_reg.fit(X_reg_train, y_reg_train)

            st.session_state["automl_reg"] = automl_reg
            st.session_state["reg_test_data"] = (X_reg_test, y_reg_test)
            st.session_state["reg_city_name"] = reg_city

        except ImportError:
            st.error(
                "**mljar-supervised is not installed.** Run: `pip install mljar-supervised` "
                "and restart the app."
            )
        except Exception as e:
            st.error(f"MLJAR encountered an error: {e}")

if "automl_reg" in st.session_state:
    automl_reg = st.session_state["automl_reg"]
    X_reg_test, y_reg_test = st.session_state["reg_test_data"]
    reg_city_name = st.session_state["reg_city_name"]

    st.subheader("MLJAR Regression Leaderboard")
    try:
        reg_leaderboard = automl_reg.get_leaderboard()
        st.dataframe(reg_leaderboard, use_container_width=True, hide_index=True)
    except Exception:
        st.info("Leaderboard display not available in this MLJAR version.")

    y_reg_pred = automl_reg.predict(X_reg_test)
    reg_rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    reg_mae = np.mean(np.abs(y_reg_test - y_reg_pred))
    reg_r2 = r2_score(y_reg_test, y_reg_pred)

    st.subheader(f"Test Set Performance — {reg_city_name}")

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{reg_rmse:.2f}°C",
                help="Average prediction error (root mean squared)")
    col2.metric("MAE", f"{reg_mae:.2f}°C",
                help="Average absolute prediction error")
    col3.metric("R-squared", f"{reg_r2:.4f}",
                help="Fraction of temperature variance explained (1.0 = perfect)")

    st.markdown(
        f"An RMSE of {reg_rmse:.2f}°C means MLJAR's best model is, on average, off by "
        f"about {reg_rmse:.1f} degrees when predicting {reg_city_name}'s temperature from "
        f"humidity, wind, pressure, hour, and month. For context, {reg_city_name}'s "
        f"temperature range across the year is roughly "
        f"{y_reg_test.max() - y_reg_test.min():.0f}°C."
    )

    # Predicted vs Actual scatter
    pred_df = pd.DataFrame({
        "Actual Temperature (°C)": y_reg_test.values,
        "Predicted Temperature (°C)": y_reg_pred.flatten() if hasattr(y_reg_pred, 'flatten') else y_reg_pred,
    })

    fig_scatter = px.scatter(
        pred_df, x="Actual Temperature (°C)", y="Predicted Temperature (°C)",
        opacity=0.3, title=f"MLJAR Predictions vs Actual — {reg_city_name}",
    )
    # Add perfect prediction line
    temp_min = min(y_reg_test.min(), pred_df["Predicted Temperature (°C)"].min())
    temp_max = max(y_reg_test.max(), pred_df["Predicted Temperature (°C)"].max())
    fig_scatter.add_trace(go.Scatter(
        x=[temp_min, temp_max], y=[temp_min, temp_max],
        mode="lines", name="Perfect prediction",
        line=dict(color="#E63946", dash="dash", width=2),
    ))
    apply_common_layout(fig_scatter, height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown(
        "Each dot is one weather reading. If MLJAR were perfect, every dot would sit on "
        "the red dashed line (predicted = actual). Points above the line are overestimates; "
        "below are underestimates. The tighter the cloud hugs the line, the better the model."
    )

    # Residual distribution
    residuals = y_reg_test.values - (y_reg_pred.flatten() if hasattr(y_reg_pred, 'flatten') else y_reg_pred)
    fig_resid = px.histogram(
        x=residuals, nbins=50,
        title=f"Residual Distribution (Actual - Predicted) — {reg_city_name}",
        labels={"x": "Residual (°C)", "y": "Count"},
    )
    apply_common_layout(fig_resid, height=400)
    st.plotly_chart(fig_resid, use_container_width=True)

    insight_box(
        f"The residuals should be centered around 0 (no systematic bias) and roughly "
        f"bell-shaped. If you see a skew or heavy tails, the model is systematically "
        f"wrong in certain conditions. Compare this RMSE ({reg_rmse:.2f}°C) to what you "
        f"achieved with polynomial regression (Ch 19) or multiple regression (Ch 18). "
        f"MLJAR likely does better because it can use non-linear models like XGBoost that "
        f"capture complex interactions between humidity, pressure, and time of day."
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: What MLJAR actually does under the hood
# ─────────────────────────────────────────────────────────────────────────────
st.header("What MLJAR Is Actually Doing Under the Hood")

st.markdown(
    "When you click 'Run,' MLJAR isn't doing anything you haven't seen in this course. "
    "It's just doing all of it, systematically, in minutes. Here's the pipeline:"
)

st.markdown(
    "**Step 1: Data Preprocessing**\n"
    "- Detects feature types (numeric, categorical, datetime)\n"
    "- Handles missing values (our data is clean, but MLJAR checks anyway)\n"
    "- Optionally engineers features (interactions, polynomial terms)\n\n"
    "**Step 2: Model Training**\n"
    "MLJAR tries a curated set of algorithms — the greatest hits of ML:\n"
    "- **Baseline**: A dummy model (most frequent class, or mean value) to set a floor\n"
    "- **Linear Model**: Logistic/linear regression with regularization (Ch 20-21)\n"
    "- **Decision Tree**: A single tree, shallow (Ch 22)\n"
    "- **Random Forest**: Bagged trees (Ch 23, 47)\n"
    "- **XGBoost**: Gradient-boosted trees (Ch 48)\n"
    "- **LightGBM**: Another gradient boosting library, often faster\n"
    "- **Neural Network**: A small feedforward network (Ch 51)\n"
    "- **Nearest Neighbors**: KNN (Ch 25)\n\n"
    "For each algorithm, it tries multiple hyperparameter configurations.\n\n"
    "**Step 3: Evaluation**\n"
    "- Uses internal cross-validation (Ch 43) to score each model honestly\n"
    "- Ranks models by validation performance\n\n"
    "**Step 4: Ensembling** (in Perform and Compete modes)\n"
    "- Builds stacking ensembles (Ch 49) from the top models\n"
    "- Uses hill climbing to find the optimal ensemble weights\n"
    "- The final model is often an ensemble of 3-5 diverse base models"
)

concept_box(
    "The Three Modes, Explained",
    "<b>Explain</b>: Tries ~5 algorithms with default hyperparameters. Fast. "
    "Good for a first look — 'is this dataset learnable at all?'<br><br>"
    "<b>Perform</b>: Tries more algorithms, tunes hyperparameters, builds ensembles. "
    "This is what you'd use in practice for most projects.<br><br>"
    "<b>Compete</b>: Throws everything at the wall. Feature engineering, stacking, "
    "hill climbing, neural architecture search. Designed to squeeze out every last "
    "fraction of a percent. Overkill for most real-world use cases, but great for "
    "competitions (hence the name).",
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Code example
# ─────────────────────────────────────────────────────────────────────────────
st.header("The Code Is Embarrassingly Simple")

st.markdown(
    "After 62 chapters of learning sklearn APIs, hyperparameter grids, cross-validation "
    "schemes, and ensemble architectures, here's what MLJAR reduces it to:"
)

code_example("""from supervised import AutoML
from sklearn.model_selection import train_test_split

# ── Classification: predict city from weather ──────────────────────
# X = DataFrame with columns: temperature_c, relative_humidity_pct,
#     wind_speed_kmh, surface_pressure_hpa
# y = Series of city names: 'Dallas', 'NYC', 'Houston', etc.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# That's it. Three lines.
automl = AutoML(mode="Perform")
automl.fit(X_train, y_train)
predictions = automl.predict(X_test)

# See what it tried
leaderboard = automl.get_leaderboard()
print(leaderboard)

# ── Regression: predict temperature ────────────────────────────────
automl_reg = AutoML(mode="Perform", ml_task="regression")
automl_reg.fit(X_train, y_train)
temp_predictions = automl_reg.predict(X_test)

# MLJAR also generates a full report in the results directory:
# - Model comparison plots
# - Feature importance for each model
# - SHAP explanations
# - Confusion matrices
""")

st.markdown(
    "Three lines to do what took us 62 chapters to learn. But those 62 chapters are "
    "why you can *interpret* what MLJAR gives you, *debug* it when it fails, and *know* "
    "when it's appropriate to use."
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: When to use AutoML vs hand-building
# ─────────────────────────────────────────────────────────────────────────────
st.header("When AutoML, When Not")

col1, col2 = st.columns(2)
with col1:
    concept_box(
        "AutoML Shines When...",
        "- You have <b>tabular data</b> (rows and columns, like our weather dataset)<br>"
        "- You want a <b>quick baseline</b> — 'how good can a model get on this data?'<br>"
        "- You're <b>comparing many approaches</b> and don't want to code each one<br>"
        "- You need <b>reproducible results</b> (MLJAR saves every model and its config)<br>"
        "- You're in a <b>Kaggle competition</b> and want to try everything fast",
    )
with col2:
    concept_box(
        "Hand-Building Is Better When...",
        "- You need <b>interpretability</b> (AutoML ensembles are hard to explain)<br>"
        "- You have <b>domain-specific constraints</b> (e.g., 'the model must use only "
        "temperature and humidity for deployment reasons')<br>"
        "- Your data needs <b>custom preprocessing</b> (time series lags, spatial features)<br>"
        "- You're building a <b>production system</b> where simplicity and speed matter<br>"
        "- You want to <b>learn</b> — which is, after all, why you're reading this course",
    )

warning_box(
    "AutoML can give you a false sense of security. A high accuracy number doesn't mean "
    "the model is right for the right reasons. Always check: Does the feature importance "
    "make physical sense? Are the predictions reasonable at the extremes? Would the model "
    "generalize to a 7th city it hasn't seen? AutoML automates the mechanics of ML, "
    "not the thinking."
)

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()

quiz(
    "MLJAR's 'Compete' mode achieves 68% accuracy on the city classification task. "
    "Your hand-built random forest from Chapter 23 achieved 64%. What should you conclude?",
    [
        "MLJAR is always better — you wasted time learning ML",
        "The 4% gain comes from trying more algorithms and building ensembles, "
        "which MLJAR automates",
        "Your random forest had a bug",
        "The dataset is too easy",
    ],
    correct_idx=1,
    explanation=(
        "MLJAR's advantage usually comes from (1) trying algorithms you didn't think of, "
        "(2) tuning hyperparameters more thoroughly, and (3) building stacking ensembles "
        "of the top models. A 4% gain from automation is typical. The point of learning "
        "ML by hand is understanding *why* the ensemble works, so you can make informed "
        "decisions about deployment, debugging, and improvement."
    ),
    key="ch63_quiz1",
)

quiz(
    "MLJAR says its best model uses 'Ensemble_Stacked' as the top entry on the "
    "leaderboard. What is this?",
    [
        "A single very large neural network",
        "A stacking ensemble (Ch 49) combining several different base models",
        "A random forest with more trees than usual",
        "A feature engineering technique",
    ],
    correct_idx=1,
    explanation=(
        "MLJAR's stacked ensemble is exactly what we built in Chapter 49: multiple diverse "
        "base models (XGBoost, LightGBM, random forest, etc.) whose predictions are combined "
        "by a meta-learner. MLJAR automates the model selection, the cross-validation for "
        "generating meta-learner training data, and the hill-climbing optimization of "
        "ensemble weights. Same concept, automated execution."
    ),
    key="ch63_quiz2",
)

quiz(
    "You run MLJAR and get 95% accuracy on your weather dataset. Should you trust it?",
    [
        "Yes — 95% is very high so the model must be good",
        "Maybe — check for data leakage, test on truly held-out data, and verify "
        "feature importance makes physical sense",
        "No — AutoML always overfits",
        "Only if it used gradient boosting",
    ],
    correct_idx=1,
    explanation=(
        "95% accuracy on a 6-city weather classification sounds suspiciously high. "
        "Before celebrating: (1) Is there data leakage? Did a datetime feature sneak in "
        "that effectively encodes the city? (2) Is the test set truly independent? "
        "(3) Does feature importance make sense — is temperature the top feature, or is "
        "some proxy variable doing the work? AutoML is powerful, but it can't detect "
        "leakage or nonsensical features. That's your job."
    ),
    key="ch63_quiz3",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "**MLJAR AutoML** automates the entire ML pipeline: algorithm selection, "
    "hyperparameter tuning, feature engineering, and ensemble building. You give it "
    "data and a target; it gives you the best model it can find.",
    "**Three modes**: Explain (fast baseline), Perform (balanced), Compete (maximum "
    "accuracy). Start with Explain, use Perform for real work.",
    "**It's not magic**: MLJAR uses the same algorithms you've learned (random forests, "
    "XGBoost, logistic regression, neural nets, stacking) — it just tries them all "
    "systematically and picks the winner.",
    "**Understanding ML is still essential**: You need to interpret results, spot "
    "leakage, validate that feature importance makes physical sense, and decide "
    "whether the model is appropriate for your use case.",
    "**AutoML as a benchmark**: Run MLJAR first to see how good a model *can* get on "
    "your data. Then hand-build models with constraints (interpretability, speed, "
    "simplicity) knowing what accuracy ceiling to expect.",
    "**Our weather data**: MLJAR will typically match or slightly beat our hand-built "
    "models, with the gain coming from ensemble stacking and broader hyperparameter search.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 62: Capstone Project",
    prev_page="62_Capstone_Project.py",
)
