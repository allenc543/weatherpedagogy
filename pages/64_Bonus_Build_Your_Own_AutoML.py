"""Bonus Chapter: Build Your Own AutoML -- A fully visual, step-by-step pipeline."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier,
    StackingClassifier, VotingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, log_loss,
)
from sklearn.inspection import permutation_importance

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.ml_helpers import plot_confusion_matrix
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

warnings.filterwarnings("ignore")

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(64, "Bonus: Build Your Own AutoML", part="Bonus")

df = load_data()
fdf = sidebar_filters(df)

# ─────────────────────────────────────────────────────────────────────────────
# PROLOGUE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "In the last bonus chapter we used MLJAR — someone else's AutoML. Now we're "
    "going to build our own, from scratch, and **watch every single step happen**. "
    "No black boxes. Every decision the pipeline makes, you see it. Every model it "
    "tries, you see the score. Every feature it engineers, you see the distribution."
)
st.markdown(
    "**The task** (same as always): Given a single hourly weather reading — "
    "temperature (°C), humidity (%), wind speed (km/h), and pressure (hPa) — "
    "predict which of our 6 cities (Dallas, San Antonio, Houston, Austin, NYC, LA) "
    "it came from. We'll build an automated pipeline that goes from raw data to a "
    "deployed-ready model in 10 visible steps."
)

run_pipeline = st.button(
    "Run the Full Pipeline (takes ~1-2 minutes)",
    type="primary", key="run_pipeline"
)

if not run_pipeline and "pipeline_results" not in st.session_state:
    st.info("Click the button above to run the full AutoML pipeline. Every step will be shown with visualizations.")
    st.stop()

# If we have cached results, use them; otherwise run the pipeline
if run_pipeline or "pipeline_results" in st.session_state:

    if run_pipeline:
        # We'll store everything in a dict
        R = {}
        progress = st.progress(0, text="Starting pipeline...")

        # ==================================================================
        # STEP 1: DATA PROFILING
        # ==================================================================
        progress.progress(5, text="Step 1/10: Profiling the raw data...")

        raw = fdf[FEATURE_COLS + ["city", "hour", "month", "day_of_year"]].dropna()

        # Subsample for speed
        sample_size = min(15000, len(raw))
        raw_sample = raw.sample(sample_size, random_state=42)

        R["raw_sample"] = raw_sample
        R["sample_size"] = sample_size
        R["n_cities"] = raw_sample["city"].nunique()
        R["city_counts"] = raw_sample["city"].value_counts()
        R["feature_stats"] = raw_sample[FEATURE_COLS].describe().T

        # ==================================================================
        # STEP 2: FEATURE ENGINEERING
        # ==================================================================
        progress.progress(15, text="Step 2/10: Engineering features...")

        eng = raw_sample.copy()

        # Cyclical encoding of hour and month
        eng["hour_sin"] = np.sin(2 * np.pi * eng["hour"] / 24)
        eng["hour_cos"] = np.cos(2 * np.pi * eng["hour"] / 24)
        eng["month_sin"] = np.sin(2 * np.pi * eng["month"] / 12)
        eng["month_cos"] = np.cos(2 * np.pi * eng["month"] / 12)
        eng["day_sin"] = np.sin(2 * np.pi * eng["day_of_year"] / 365)
        eng["day_cos"] = np.cos(2 * np.pi * eng["day_of_year"] / 365)

        # Interaction features
        eng["temp_x_humidity"] = eng["temperature_c"] * eng["relative_humidity_pct"]
        eng["wind_x_pressure"] = eng["wind_speed_kmh"] * eng["surface_pressure_hpa"]
        eng["temp_minus_dewpoint"] = eng["temperature_c"] - (eng["temperature_c"] - (100 - eng["relative_humidity_pct"]) / 5)

        # Polynomial features for temperature
        eng["temp_squared"] = eng["temperature_c"] ** 2

        all_features = FEATURE_COLS + [
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "day_sin", "day_cos",
            "temp_x_humidity", "wind_x_pressure", "temp_minus_dewpoint",
            "temp_squared",
        ]

        R["eng"] = eng
        R["all_features"] = all_features
        R["n_original"] = len(FEATURE_COLS)
        R["n_engineered"] = len(all_features)

        # ==================================================================
        # STEP 3: TRAIN/TEST SPLIT
        # ==================================================================
        progress.progress(20, text="Step 3/10: Splitting data...")

        le = LabelEncoder()
        X = eng[all_features]
        y = le.fit_transform(eng["city"])
        city_labels = le.classes_

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        R["X_train"] = X_train
        R["X_test"] = X_test
        R["y_train"] = y_train
        R["y_test"] = y_test
        R["city_labels"] = city_labels
        R["le"] = le

        # ==================================================================
        # STEP 4: SCALING COMPARISON
        # ==================================================================
        progress.progress(25, text="Step 4/10: Comparing scalers...")

        scalers = {
            "No Scaling": None,
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
        }

        scaler_results = {}
        quick_model = LogisticRegression(max_iter=500, random_state=42)

        for name, scaler in scalers.items():
            if scaler is not None:
                X_tr_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train), columns=all_features, index=X_train.index
                )
                X_te_scaled = pd.DataFrame(
                    scaler.transform(X_test), columns=all_features, index=X_test.index
                )
            else:
                X_tr_scaled = X_train
                X_te_scaled = X_test

            quick_model.fit(X_tr_scaled, y_train)
            scaler_results[name] = {
                "train_acc": accuracy_score(y_train, quick_model.predict(X_tr_scaled)),
                "test_acc": accuracy_score(y_test, quick_model.predict(X_te_scaled)),
            }

        best_scaler_name = max(scaler_results, key=lambda k: scaler_results[k]["test_acc"])
        best_scaler = scalers[best_scaler_name]

        # Apply best scaler
        if best_scaler is not None:
            best_scaler_obj = type(best_scaler)()  # fresh instance
            X_train_scaled = pd.DataFrame(
                best_scaler_obj.fit_transform(X_train), columns=all_features, index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                best_scaler_obj.transform(X_test), columns=all_features, index=X_test.index
            )
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            best_scaler_obj = None

        R["scaler_results"] = scaler_results
        R["best_scaler_name"] = best_scaler_name
        R["X_train_scaled"] = X_train_scaled
        R["X_test_scaled"] = X_test_scaled

        # ==================================================================
        # STEP 5: MODEL ZOO — TRY EVERYTHING
        # ==================================================================
        progress.progress(30, text="Step 5/10: Training 10 different models...")

        model_zoo = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree (d=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Decision Tree (d=15)": DecisionTreeClassifier(max_depth=15, random_state=42),
            "Random Forest (50 trees)": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
            "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42, algorithm="SAMME"),
            "KNN (k=7)": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
            "Naive Bayes": GaussianNB(),
        }

        model_results = {}
        fitted_models = {}

        for i, (name, model) in enumerate(model_zoo.items()):
            pct = 30 + int(40 * (i + 1) / len(model_zoo))
            progress.progress(pct, text=f"Step 5/10: Training {name}...")

            start_t = time.time()
            model.fit(X_train_scaled, y_train)
            train_time = time.time() - start_t

            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)

            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5,
                                        scoring="accuracy", n_jobs=-1)

            model_results[name] = {
                "train_acc": accuracy_score(y_train, train_pred),
                "test_acc": accuracy_score(y_test, test_pred),
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "train_time": train_time,
                "cv_scores": cv_scores,
                "test_pred": test_pred,
            }
            fitted_models[name] = model

        R["model_results"] = model_results
        R["fitted_models"] = fitted_models

        # ==================================================================
        # STEP 6: CROSS-VALIDATION DEEP DIVE (top 3 models)
        # ==================================================================
        progress.progress(75, text="Step 6/10: Deep cross-validation on top models...")

        sorted_models = sorted(model_results.items(), key=lambda x: x[1]["cv_mean"], reverse=True)
        top_3_names = [name for name, _ in sorted_models[:3]]

        cv_details = {}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name in top_3_names:
            fold_results = []
            for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
                X_fold_tr = X_train_scaled.iloc[tr_idx]
                y_fold_tr = y_train[tr_idx]
                X_fold_val = X_train_scaled.iloc[val_idx]
                y_fold_val = y_train[val_idx]

                m = type(fitted_models[name])(**fitted_models[name].get_params())
                m.fit(X_fold_tr, y_fold_tr)
                val_acc = accuracy_score(y_fold_val, m.predict(X_fold_val))
                fold_results.append({"fold": fold_idx + 1, "accuracy": val_acc})

            cv_details[name] = pd.DataFrame(fold_results)

        R["top_3_names"] = top_3_names
        R["cv_details"] = cv_details

        # ==================================================================
        # STEP 7: HYPERPARAMETER SENSITIVITY (top model)
        # ==================================================================
        progress.progress(80, text="Step 7/10: Hyperparameter sensitivity analysis...")

        best_model_name = top_3_names[0]
        hp_results = []

        # Do a simple sweep on the best model type
        if "Forest" in best_model_name or "Extra" in best_model_name:
            for n_est in [10, 25, 50, 100, 200]:
                for depth in [5, 10, 15, 20, None]:
                    if "Extra" in best_model_name:
                        m = ExtraTreesClassifier(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1)
                    else:
                        m = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1)
                    cv = cross_val_score(m, X_train_scaled, y_train, cv=3, scoring="accuracy", n_jobs=-1)
                    hp_results.append({
                        "n_estimators": n_est,
                        "max_depth": str(depth) if depth else "None",
                        "cv_accuracy": cv.mean(),
                    })
        elif "Gradient" in best_model_name:
            for lr in [0.01, 0.05, 0.1, 0.2]:
                for n_est in [50, 100, 200]:
                    for depth in [2, 3, 5]:
                        m = GradientBoostingClassifier(n_estimators=n_est, max_depth=depth, learning_rate=lr, random_state=42)
                        cv = cross_val_score(m, X_train_scaled, y_train, cv=3, scoring="accuracy", n_jobs=-1)
                        hp_results.append({
                            "learning_rate": lr,
                            "n_estimators": n_est,
                            "max_depth": depth,
                            "cv_accuracy": cv.mean(),
                        })
        else:
            # Generic: just report the default
            hp_results.append({"config": "default", "cv_accuracy": model_results[best_model_name]["cv_mean"]})

        hp_df = pd.DataFrame(hp_results)
        R["hp_df"] = hp_df
        R["best_model_name"] = best_model_name

        # Find best hyperparams and retrain
        if len(hp_df) > 1:
            best_hp_idx = hp_df["cv_accuracy"].idxmax()
            best_hp_row = hp_df.iloc[best_hp_idx]
            R["best_hp_row"] = best_hp_row

            if "n_estimators" in hp_df.columns and "max_depth" in hp_df.columns and "learning_rate" not in hp_df.columns:
                best_n = int(best_hp_row["n_estimators"])
                best_d = None if best_hp_row["max_depth"] == "None" else int(best_hp_row["max_depth"])
                if "Extra" in best_model_name:
                    tuned_model = ExtraTreesClassifier(n_estimators=best_n, max_depth=best_d, random_state=42, n_jobs=-1)
                else:
                    tuned_model = RandomForestClassifier(n_estimators=best_n, max_depth=best_d, random_state=42, n_jobs=-1)
            elif "learning_rate" in hp_df.columns:
                tuned_model = GradientBoostingClassifier(
                    n_estimators=int(best_hp_row["n_estimators"]),
                    max_depth=int(best_hp_row["max_depth"]),
                    learning_rate=best_hp_row["learning_rate"],
                    random_state=42,
                )
            else:
                tuned_model = fitted_models[best_model_name]
        else:
            tuned_model = fitted_models[best_model_name]

        tuned_model.fit(X_train_scaled, y_train)
        tuned_test_acc = accuracy_score(y_test, tuned_model.predict(X_test_scaled))
        R["tuned_model"] = tuned_model
        R["tuned_test_acc"] = tuned_test_acc

        # ==================================================================
        # STEP 8: ENSEMBLE BUILDING
        # ==================================================================
        progress.progress(85, text="Step 8/10: Building ensemble from top models...")

        # Build stacking ensemble from top 3
        estimators = [(name, fitted_models[name]) for name in top_3_names]

        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5, n_jobs=-1,
        )
        stack.fit(X_train_scaled, y_train)
        stack_test_acc = accuracy_score(y_test, stack.predict(X_test_scaled))
        stack_pred = stack.predict(X_test_scaled)

        # Also try voting
        vote = VotingClassifier(estimators=estimators, voting="hard", n_jobs=-1)
        vote.fit(X_train_scaled, y_train)
        vote_test_acc = accuracy_score(y_test, vote.predict(X_test_scaled))

        R["stack_test_acc"] = stack_test_acc
        R["stack_pred"] = stack_pred
        R["vote_test_acc"] = vote_test_acc
        R["stack_model"] = stack

        # ==================================================================
        # STEP 9: FINAL EVALUATION
        # ==================================================================
        progress.progress(90, text="Step 9/10: Final evaluation...")

        # Pick the overall best
        final_candidates = {
            f"Tuned {best_model_name}": tuned_test_acc,
            "Stacking Ensemble": stack_test_acc,
            "Voting Ensemble": vote_test_acc,
        }
        overall_best_name = max(final_candidates, key=final_candidates.get)
        overall_best_acc = final_candidates[overall_best_name]

        if overall_best_name == "Stacking Ensemble":
            final_pred = stack_pred
            final_model = stack
        elif overall_best_name == "Voting Ensemble":
            final_pred = vote.predict(X_test_scaled)
            final_model = vote
        else:
            final_pred = tuned_model.predict(X_test_scaled)
            final_model = tuned_model

        R["final_candidates"] = final_candidates
        R["overall_best_name"] = overall_best_name
        R["overall_best_acc"] = overall_best_acc
        R["final_pred"] = final_pred
        R["final_model"] = final_model

        # Per-class metrics
        prec, rec, f1, sup = precision_recall_fscore_support(y_test, final_pred, labels=range(len(city_labels)))
        per_class = pd.DataFrame({
            "City": city_labels,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "Support": sup.astype(int),
        })
        R["per_class"] = per_class
        R["cm"] = confusion_matrix(y_test, final_pred)

        # ==================================================================
        # STEP 10: FEATURE IMPORTANCE
        # ==================================================================
        progress.progress(95, text="Step 10/10: Computing feature importance...")

        # Use the tuned model for importance (needs to be a single model, not stack)
        if hasattr(tuned_model, "feature_importances_"):
            imp = tuned_model.feature_importances_
            imp_type = "MDI (built-in)"
        else:
            # Permutation importance
            perm = permutation_importance(tuned_model, X_test_scaled, y_test,
                                          n_repeats=5, random_state=42, n_jobs=-1)
            imp = perm.importances_mean
            imp_type = "Permutation"

        imp_df = pd.DataFrame({
            "Feature": all_features,
            "Importance": imp,
        }).sort_values("Importance", ascending=False)
        R["imp_df"] = imp_df
        R["imp_type"] = imp_type

        progress.progress(100, text="Pipeline complete!")
        time.sleep(0.5)
        progress.empty()

        st.session_state["pipeline_results"] = R

    # ──────────────────────────────────────────────────────────────────────
    # DISPLAY ALL RESULTS
    # ──────────────────────────────────────────────────────────────────────
    R = st.session_state["pipeline_results"]

    st.success("Pipeline complete! Scroll through all 10 steps below.")
    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1 DISPLAY: Data Profiling
    # ══════════════════════════════════════════════════════════════════════
    st.header("Step 1: Data Profiling — What Are We Working With?")

    st.markdown(
        f"Before building any model, we need to understand the raw data. We sampled "
        f"**{R['sample_size']:,}** readings from the filtered dataset. Here's what "
        f"each of the 4 weather features looks like:"
    )

    # Distribution of each feature
    fig_dist = make_subplots(rows=2, cols=2, subplot_titles=[
        FEATURE_LABELS.get(f, f) for f in FEATURE_COLS
    ])

    colors = ["#E63946", "#2A9D8F", "#264653", "#FB8500"]
    for i, feat in enumerate(FEATURE_COLS):
        row, col = (i // 2) + 1, (i % 2) + 1
        fig_dist.add_trace(
            go.Histogram(x=R["raw_sample"][feat], nbinsx=50,
                         marker_color=colors[i], opacity=0.7, name=FEATURE_LABELS.get(feat, feat)),
            row=row, col=col,
        )
    fig_dist.update_layout(height=500, showlegend=False, template="plotly_white",
                            title_text="Raw Feature Distributions")
    st.plotly_chart(fig_dist, use_container_width=True)

    # Class balance
    fig_balance = px.bar(
        x=R["city_counts"].index, y=R["city_counts"].values,
        color=R["city_counts"].index, color_discrete_map=CITY_COLORS,
        title="Class Balance: How Many Readings Per City?",
        labels={"x": "City", "y": "Count"},
    )
    apply_common_layout(fig_balance, height=350)
    st.plotly_chart(fig_balance, use_container_width=True)

    st.markdown(
        f"**{R['n_cities']} cities**, roughly balanced. The distributions show: "
        f"temperature spans a wide range (cold NYC winters to hot Texas summers), "
        f"humidity clusters around 50-80%, wind speed is right-skewed (mostly calm, "
        f"occasionally gusty), and pressure is tightly concentrated around 1010-1020 hPa."
    )

    # Feature stats table
    st.markdown("**Summary statistics** (mean, std, min/max):")
    stats_display = R["feature_stats"][["mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats_display.index = [FEATURE_LABELS.get(f, f) for f in stats_display.index]
    st.dataframe(stats_display.style.format("{:.2f}"), use_container_width=True)

    insight_box(
        "Notice the scale differences: temperature is 0-40°C, pressure is 990-1040 hPa. "
        "That 1000x difference will wreck distance-based models like KNN. We'll need scaling."
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2 DISPLAY: Feature Engineering
    # ══════════════════════════════════════════════════════════════════════
    st.header("Step 2: Feature Engineering — Creating New Signal")

    st.markdown(
        f"We started with **{R['n_original']} raw features**. Our pipeline "
        f"engineered **{R['n_engineered'] - R['n_original']} new features**, "
        f"bringing the total to **{R['n_engineered']}**:"
    )

    eng_table = pd.DataFrame({
        "Feature": R["all_features"],
        "Type": (
            ["Original"] * 4 +
            ["Cyclical (hour)"] * 2 + ["Cyclical (month)"] * 2 + ["Cyclical (day)"] * 2 +
            ["Interaction"] * 3 + ["Polynomial"] * 1
        ),
        "Description": [
            "Raw temperature (°C)", "Raw humidity (%)", "Raw wind speed (km/h)", "Raw pressure (hPa)",
            "sin(2π·hour/24)", "cos(2π·hour/24)",
            "sin(2π·month/12)", "cos(2π·month/12)",
            "sin(2π·day/365)", "cos(2π·day/365)",
            "temp × humidity", "wind × pressure",
            "100 - humidity (dew point proxy)",
            "temperature²",
        ],
    })
    st.dataframe(eng_table, use_container_width=True, hide_index=True)

    # Show cyclical features
    fig_cyc = make_subplots(rows=1, cols=2, subplot_titles=["Hour: Sin vs Cos", "Month: Sin vs Cos"])
    fig_cyc.add_trace(
        go.Scatter(x=R["eng"]["hour_sin"], y=R["eng"]["hour_cos"],
                   mode="markers", marker=dict(size=2, opacity=0.2, color="#2A9D8F"), name="Hour"),
        row=1, col=1,
    )
    fig_cyc.add_trace(
        go.Scatter(x=R["eng"]["month_sin"], y=R["eng"]["month_cos"],
                   mode="markers", marker=dict(size=2, opacity=0.2, color="#E63946"), name="Month"),
        row=1, col=2,
    )
    fig_cyc.update_layout(height=350, template="plotly_white",
                           title_text="Cyclical Features Form Circles (That's the Point)")
    st.plotly_chart(fig_cyc, use_container_width=True)

    st.markdown(
        "The cyclical encoding maps hours and months onto circles — hour 23 is now "
        "close to hour 0 (as it should be), and December is close to January. "
        "Without this, a linear model would think December (12) and January (1) are "
        "11 months apart."
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3 DISPLAY: Train/Test Split
    # ══════════════════════════════════════════════════════════════════════
    st.header("Step 3: Train/Test Split — Drawing the Line")

    col1, col2, col3 = st.columns(3)
    col1.metric("Training Set", f"{len(R['X_train']):,} readings")
    col2.metric("Test Set", f"{len(R['X_test']):,} readings")
    col3.metric("Split Ratio", "80% / 20%")

    st.markdown(
        "We used **stratified** splitting — each city has the same proportion in train "
        "and test. This prevents the test set from accidentally having too many NYC "
        "readings and too few Austin readings."
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4 DISPLAY: Scaling Comparison
    # ══════════════════════════════════════════════════════════════════════
    st.header("Step 4: Scaling Comparison — Does It Matter?")

    st.markdown(
        "We trained a quick logistic regression with three different scaling strategies "
        "to see which works best on our data:"
    )

    scaler_df = pd.DataFrame([
        {"Scaler": name, "Train Accuracy": res["train_acc"], "Test Accuracy": res["test_acc"]}
        for name, res in R["scaler_results"].items()
    ]).sort_values("Test Accuracy", ascending=False)

    fig_scaler = px.bar(scaler_df, x="Scaler", y="Test Accuracy",
                         color="Scaler", color_discrete_sequence=["#2A9D8F", "#E63946", "#264653"],
                         title="Scaling Impact on Logistic Regression Accuracy")
    apply_common_layout(fig_scaler, height=350)
    st.plotly_chart(fig_scaler, use_container_width=True)

    st.dataframe(scaler_df.style.format({"Train Accuracy": "{:.4f}", "Test Accuracy": "{:.4f}"}),
                 use_container_width=True, hide_index=True)

    st.markdown(f"**Winner: {R['best_scaler_name']}** — used for all subsequent models.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5 DISPLAY: Model Zoo
    # ══════════════════════════════════════════════════════════════════════
    st.header("Step 5: The Model Zoo — Try Everything")

    st.markdown(
        "This is the core of AutoML: throw 10 different algorithms at the data and "
        "see which ones stick. Each model gets the same scaled training data and is "
        "evaluated with 5-fold cross-validation."
    )

    zoo_df = pd.DataFrame([
        {
            "Model": name,
            "Train Acc": res["train_acc"],
            "Test Acc": res["test_acc"],
            "CV Mean": res["cv_mean"],
            "CV Std": res["cv_std"],
            "Train Time (s)": res["train_time"],
        }
        for name, res in R["model_results"].items()
    ]).sort_values("CV Mean", ascending=False)

    # Leaderboard chart
    fig_zoo = px.bar(zoo_df, x="Model", y="CV Mean",
                      error_y="CV Std",
                      color="CV Mean", color_continuous_scale="Viridis",
                      title="Model Zoo Leaderboard (5-Fold CV Accuracy)")
    apply_common_layout(fig_zoo, height=500)
    fig_zoo.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_zoo, use_container_width=True)

    st.dataframe(
        zoo_df.style.format({
            "Train Acc": "{:.4f}", "Test Acc": "{:.4f}",
            "CV Mean": "{:.4f}", "CV Std": "{:.4f}",
            "Train Time (s)": "{:.3f}",
        }).highlight_max(subset=["CV Mean"], color="#d4edda"),
        use_container_width=True, hide_index=True,
    )

    # Overfitting detector
    zoo_df["Overfit Gap"] = zoo_df["Train Acc"] - zoo_df["Test Acc"]
    fig_overfit = px.scatter(zoo_df, x="Train Acc", y="Test Acc", size="Train Time (s)",
                              text="Model", color="Overfit Gap",
                              color_continuous_scale="RdYlGn_r",
                              title="Overfitting Detector: Train vs Test Accuracy")
    fig_overfit.add_trace(go.Scatter(
        x=[0.3, 1.0], y=[0.3, 1.0], mode="lines",
        line=dict(dash="dash", color="gray"), name="Perfect generalization",
    ))
    apply_common_layout(fig_overfit, height=500)
    fig_overfit.update_traces(textposition="top center", selector=dict(mode="markers+text"))
    st.plotly_chart(fig_overfit, use_container_width=True)

    st.markdown(
        "Points on the dashed line generalize perfectly (train = test). Points above "
        "the line are overfitting (train >> test). The color shows the gap: green = "
        "good generalization, red = overfitting."
    )

    insight_box(
        f"The top 3 models are: **{R['top_3_names'][0]}**, **{R['top_3_names'][1]}**, "
        f"and **{R['top_3_names'][2]}**. These will go forward to ensemble building."
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6 DISPLAY: Cross-Validation Deep Dive
    # ══════════════════════════════════════════════════════════════════════
    st.header("Step 6: Cross-Validation Deep Dive — Is the Score Stable?")

    st.markdown(
        "A model's average CV score could hide wild fold-to-fold variation. Let's see "
        "how stable our top 3 models are across the 5 folds:"
    )

    fig_cv = go.Figure()
    cv_colors = ["#2A9D8F", "#E63946", "#264653"]
    for i, name in enumerate(R["top_3_names"]):
        fold_df = R["cv_details"][name]
        fig_cv.add_trace(go.Scatter(
            x=fold_df["fold"], y=fold_df["accuracy"],
            mode="lines+markers", name=name,
            line=dict(color=cv_colors[i], width=2), marker=dict(size=8),
        ))
    apply_common_layout(fig_cv, title="Fold-by-Fold Accuracy for Top 3 Models", height=400)
    fig_cv.update_layout(xaxis_title="Fold", yaxis_title="Accuracy", xaxis=dict(dtick=1))
    st.plotly_chart(fig_cv, use_container_width=True)

    st.markdown(
        "Ideally, each model's line should be flat (consistent across folds). "
        "Big swings mean the model is sensitive to which data it trains on — a sign "
        "of high variance."
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 7 DISPLAY: Hyperparameter Sensitivity
    # ══════════════════════════════════════════════════════════════════════
    st.header("Step 7: Hyperparameter Tuning — Finding the Sweet Spot")

    st.markdown(
        f"We took the top model (**{R['best_model_name']}**) and swept across its "
        f"key hyperparameters to find the best configuration."
    )

    hp_df = R["hp_df"]

    if len(hp_df) > 1 and "n_estimators" in hp_df.columns and "max_depth" in hp_df.columns and "learning_rate" not in hp_df.columns:
        # Forest-type: heatmap of n_estimators vs max_depth
        pivot = hp_df.pivot_table(values="cv_accuracy", index="max_depth", columns="n_estimators")
        fig_hp = go.Figure(data=go.Heatmap(
            z=pivot.values, x=[str(c) for c in pivot.columns],
            y=[str(r) for r in pivot.index],
            colorscale="Viridis", text=np.round(pivot.values, 4), texttemplate="%{text}",
        ))
        apply_common_layout(fig_hp, title=f"Hyperparameter Grid: {R['best_model_name']}", height=400)
        fig_hp.update_layout(xaxis_title="n_estimators", yaxis_title="max_depth")
        st.plotly_chart(fig_hp, use_container_width=True)
    elif len(hp_df) > 1 and "learning_rate" in hp_df.columns:
        fig_hp = px.scatter_3d(hp_df, x="learning_rate", y="n_estimators", z="cv_accuracy",
                                color="cv_accuracy", color_continuous_scale="Viridis",
                                size="max_depth", title=f"Hyperparameter Space: {R['best_model_name']}")
        st.plotly_chart(fig_hp, use_container_width=True)
    else:
        st.dataframe(hp_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    col1.metric("Before Tuning", f"{R['model_results'][R['best_model_name']]['test_acc']:.4f}")
    col2.metric("After Tuning", f"{R['tuned_test_acc']:.4f}",
                delta=f"{R['tuned_test_acc'] - R['model_results'][R['best_model_name']]['test_acc']:+.4f}")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 8 DISPLAY: Ensemble Building
    # ══════════════════════════════════════════════════════════════════════
    st.header("Step 8: Ensemble Building — Combining the Best")

    st.markdown(
        f"We took the top 3 models and combined them two ways: **hard voting** "
        f"(majority rules) and **stacking** (a meta-learner decides)."
    )

    ensemble_df = pd.DataFrame({
        "Method": [f"Tuned {R['best_model_name']}", "Hard Voting", "Stacking Ensemble"],
        "Test Accuracy": [R["tuned_test_acc"], R["vote_test_acc"], R["stack_test_acc"]],
        "Type": ["Single Model", "Voting", "Stacking"],
    }).sort_values("Test Accuracy", ascending=False)

    fig_ens = px.bar(ensemble_df, x="Method", y="Test Accuracy", color="Type",
                      color_discrete_map={"Single Model": "#264653", "Voting": "#2A9D8F", "Stacking": "#E63946"},
                      title="Single Model vs Ensemble Comparison")
    apply_common_layout(fig_ens, height=400)
    st.plotly_chart(fig_ens, use_container_width=True)

    st.dataframe(ensemble_df.style.format({"Test Accuracy": "{:.4f}"}),
                 use_container_width=True, hide_index=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 9 DISPLAY: Final Evaluation
    # ══════════════════════════════════════════════════════════════════════
    st.header("Step 9: Final Evaluation — The Verdict")

    st.markdown(
        f"**The pipeline chose: {R['overall_best_name']}** with "
        f"**{R['overall_best_acc']:.4f}** test accuracy ({R['overall_best_acc']:.1%})."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", R["overall_best_name"])
    col2.metric("Test Accuracy", f"{R['overall_best_acc']:.4f}")
    col3.metric("vs Random Guessing", f"+{R['overall_best_acc'] - 1/6:.4f}",
                help="Random guessing would give 16.7% for 6 classes")

    # Confusion matrix
    st.subheader("Confusion Matrix — Where Does It Struggle?")
    fig_cm = plot_confusion_matrix(R["cm"], R["city_labels"])
    fig_cm.update_layout(title=f"Confusion Matrix: {R['overall_best_name']}")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown(
        "Read row by row: each row is the true city, each column is the prediction. "
        "Big numbers on the diagonal = correct. Off-diagonal = confusion. The biggest "
        "off-diagonal numbers show which city pairs are hardest to distinguish."
    )

    # Per-class metrics
    st.subheader("Per-City Performance")
    fig_class = make_subplots(rows=1, cols=3, subplot_titles=["Precision", "Recall", "F1 Score"])
    for i, metric in enumerate(["Precision", "Recall", "F1 Score"]):
        fig_class.add_trace(
            go.Bar(x=R["per_class"]["City"], y=R["per_class"][metric],
                   marker_color=[CITY_COLORS.get(c, "#999") for c in R["per_class"]["City"]],
                   name=metric),
            row=1, col=i+1,
        )
    fig_class.update_layout(height=350, template="plotly_white", showlegend=False)
    st.plotly_chart(fig_class, use_container_width=True)

    st.dataframe(
        R["per_class"].style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1 Score": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown(
        "Cities with distinct climates (NYC, LA) should have high precision and recall. "
        "Texas cities (Dallas, SA, Houston, Austin) will be harder because their weather "
        "genuinely overlaps."
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 10 DISPLAY: Feature Importance
    # ══════════════════════════════════════════════════════════════════════
    st.header("Step 10: Feature Importance — What Did the Model Actually Learn?")

    st.markdown(
        f"Using **{R['imp_type']}** importance from the tuned model, here's what "
        f"the model relies on most to distinguish cities:"
    )

    fig_imp = px.bar(R["imp_df"].head(14), x="Importance", y="Feature",
                      orientation="h", color="Importance",
                      color_continuous_scale="Viridis",
                      title="Feature Importance (Top 14)")
    apply_common_layout(fig_imp, height=500)
    fig_imp.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_imp, use_container_width=True)

    top_feat = R["imp_df"].iloc[0]["Feature"]
    st.markdown(
        f"**{FEATURE_LABELS.get(top_feat, top_feat)}** is the most important feature. "
        f"This makes physical sense: temperature varies dramatically across our cities "
        f"(NYC winters vs LA's mild year-round climate vs Texas heat). The engineered "
        f"features like cyclical encodings and interactions add genuine signal — the "
        f"pipeline was right to create them."
    )

    insight_box(
        "Compare this to the SHAP analysis from Chapter 42. The features that matter "
        "here should be similar: temperature dominates, humidity helps distinguish "
        "Houston (humid) from LA (dry), and time features capture when cities diverge "
        "most (winter separates NYC from everyone else)."
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # PIPELINE SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    st.header("Pipeline Summary — Everything We Did")

    summary_steps = pd.DataFrame({
        "Step": [f"Step {i}" for i in range(1, 11)],
        "Name": [
            "Data Profiling", "Feature Engineering", "Train/Test Split",
            "Scaling Comparison", "Model Zoo (10 models)", "Cross-Validation Deep Dive",
            "Hyperparameter Tuning", "Ensemble Building", "Final Evaluation",
            "Feature Importance",
        ],
        "Key Decision": [
            f"{R['sample_size']:,} readings, 6 cities, 4 raw features",
            f"4 → {R['n_engineered']} features (cyclical, interactions, polynomial)",
            "80/20 stratified split",
            f"Best scaler: {R['best_scaler_name']}",
            f"Best of 10: {R['top_3_names'][0]} (CV={R['model_results'][R['top_3_names'][0]]['cv_mean']:.4f})",
            "Verified stability across 5 folds",
            f"Tuned {R['best_model_name']}: {R['tuned_test_acc']:.4f}",
            f"Stacking: {R['stack_test_acc']:.4f}, Voting: {R['vote_test_acc']:.4f}",
            f"Winner: {R['overall_best_name']} ({R['overall_best_acc']:.4f})",
            f"Top feature: {FEATURE_LABELS.get(top_feat, top_feat)}",
        ],
    })

    st.dataframe(summary_steps, use_container_width=True, hide_index=True)

    st.markdown(
        f"From raw weather readings to a tuned, ensembled model with "
        f"**{R['overall_best_acc']:.1%} accuracy** — and you saw every decision along the way. "
        f"That's AutoML, demystified."
    )

# ── Code ─────────────────────────────────────────────────────────────────────
st.divider()

code_example("""import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load & profile
X = df[['temperature_c', 'relative_humidity_pct', 'wind_speed_kmh', 'surface_pressure_hpa']]
y = df['city']

# Step 2: Feature engineering
X['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
X['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
X['temp_x_humidity'] = X['temperature_c'] * X['relative_humidity_pct']

# Step 3: Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Step 4: Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Step 5: Try many models
models = {
    'RF': RandomForestClassifier(n_estimators=100),
    'GB': GradientBoostingClassifier(n_estimators=100),
    'ET': ExtraTreesClassifier(n_estimators=100),
}
for name, model in models.items():
    cv = cross_val_score(model, X_train_s, y_train, cv=5)
    print(f'{name}: CV={cv.mean():.4f} +/- {cv.std():.4f}')

# Step 8: Stack the top models
stack = StackingClassifier(
    estimators=list(models.items()),
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
)
stack.fit(X_train_s, y_train)
print(f'Stacking: {stack.score(X_test_s, y_test):.4f}')
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()

quiz(
    "Your AutoML pipeline's best single model gets 62% accuracy. The stacking ensemble "
    "gets 63%. Is the ensemble worth the added complexity?",
    [
        "Yes — always use the highest accuracy model",
        "Maybe — 1% gain might not justify the complexity for deployment",
        "No — 1% is within noise",
        "Yes — ensembles are always better",
    ],
    correct_idx=1,
    explanation=(
        "In production, model complexity matters. A stacking ensemble is harder to "
        "deploy, debug, and explain than a single model. A 1% gain might not be worth "
        "it if interpretability, latency, or maintenance cost matters. But in a Kaggle "
        "competition, you'd take every fraction of a percent. Context determines the answer."
    ),
    key="ch64_quiz1",
)

quiz(
    "The pipeline tries 10 models and the decision tree gets 95% train accuracy but "
    "52% test accuracy. What does the overfitting detector chart show?",
    [
        "The point sits right on the diagonal line",
        "The point sits far above the diagonal — high train, low test = overfitting",
        "The point sits below the diagonal",
        "The point disappears from the chart",
    ],
    correct_idx=1,
    explanation=(
        "The overfitting detector plots train accuracy (x) vs test accuracy (y). "
        "The diagonal line is where train = test (perfect generalization). A point "
        "at (0.95, 0.52) sits far above this line — the model has memorized the "
        "training data but learned almost nothing generalizable. The pipeline would "
        "rank this model low on the leaderboard based on CV score."
    ),
    key="ch64_quiz2",
)

quiz(
    "The feature importance chart shows 'temp_x_humidity' (an interaction feature) "
    "as the 3rd most important feature. What does this tell you?",
    [
        "Temperature and humidity are redundant — drop one",
        "The interaction between temperature and humidity carries information that "
        "neither feature alone captures",
        "The feature engineering step was unnecessary",
        "The model is overfitting to engineered features",
    ],
    correct_idx=1,
    explanation=(
        "High importance for an interaction feature means the *combination* of "
        "temperature and humidity matters beyond their individual effects. This makes "
        "physical sense: Houston is both hot AND humid, while LA is warm but dry. "
        "The product temp × humidity captures 'hot-and-humid' as a single number "
        "that helps distinguish Houston from other warm cities."
    ),
    key="ch64_quiz3",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "**AutoML is just automation of what you already know**: data profiling (Ch 1-2), "
    "feature engineering (Ch 39), scaling (Ch 40), model training (Ch 21-26), "
    "cross-validation (Ch 43), hyperparameter tuning, ensembling (Ch 47-49).",
    "**Feature engineering matters**: going from 4 raw features to 14 engineered "
    "features (cyclical time, interactions, polynomials) often gives more improvement "
    "than switching algorithms.",
    "**The overfitting detector** (train vs test scatter) instantly shows which models "
    "are memorizing vs generalizing. Points far from the diagonal are suspect.",
    "**Hyperparameter tuning usually gives modest gains** (1-3%) over defaults. "
    "The algorithm choice matters more than the hyperparameters.",
    "**Ensembles are diminishing returns**: stacking the top 3 models might beat any "
    "single model by 1-2%, but adds complexity. Worth it for competitions, often not "
    "for production.",
    "**Feature importance is your sanity check**: if the model relies heavily on "
    "features that don't make physical sense, something is wrong — regardless of accuracy.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 63: MLJAR AutoML",
    prev_page="63_Bonus_MLJAR_AutoML.py",
)
