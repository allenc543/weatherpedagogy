"""Chapter 62: Capstone Project — End-to-end workflow with three guided projects."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, scatter_chart
from utils.ml_helpers import (
    prepare_classification_data, classification_metrics, plot_confusion_matrix,
    prepare_regression_data, regression_metrics,
)
from utils.ui_components import (
    chapter_header, concept_box, insight_box, warning_box,
    code_example, takeaways, navigation,
)
from utils.constants import CITY_LIST, CITY_COLORS, FEATURE_COLS, FEATURE_LABELS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
df = load_data()
fdf = sidebar_filters(df)

chapter_header(62, "Capstone Project", part="XV")

st.markdown(
    "Over the past 61 chapters, we have built up a toolkit: descriptive statistics, "
    "visualizations, hypothesis tests, regression, classification, clustering, time series "
    "forecasting, neural networks, Bayesian inference, anomaly detection, and causal "
    "reasoning. Each chapter introduced one tool in isolation. That is like learning to "
    "use a hammer, a saw, and a level separately -- useful, but nobody hires a carpenter "
    "who has never actually built a table."
)
st.markdown(
    "This chapter is where we build the table. Three complete projects, each one a "
    "realistic end-to-end data science workflow using our 105,264 hourly weather readings "
    "from 6 cities. The question is not 'can you run a Random Forest?' -- you proved that "
    "in Chapter 46. The question is: can you pick the right tool for the right problem, "
    "prepare data correctly, establish baselines, evaluate honestly, and interpret results "
    "in a way that would survive a skeptical colleague's questions?"
)

concept_box(
    "The Data Science Pipeline (For Real This Time)",
    "Every project below follows the same pipeline, but the emphasis shifts depending "
    "on the problem:<br><br>"
    "1. <b>Problem definition</b> -- what exactly are we predicting, and what would a "
    "useful answer look like? For the city classifier, 'useful' means better than random "
    "guessing (16.7% for 6 cities). For the temperature forecast, 'useful' means better "
    "than 'tomorrow equals today.'<br><br>"
    "2. <b>Data exploration</b> -- what does the data actually look like? We have ~17,500 "
    "hourly readings per city. Are the city distributions different enough to classify? "
    "Is temperature autocorrelated enough to forecast?<br><br>"
    "3. <b>Model building</b> -- this is the part everyone fixates on, and it is maybe "
    "20% of the actual work. Choosing features and establishing baselines matters far more "
    "than choosing between model architectures.<br><br>"
    "4. <b>Evaluation</b> -- this is the part that separates real data science from "
    "cargo-cult data science. Not just 'what is the accuracy?' but 'what kinds of errors "
    "does the model make, and do those errors make sense physically?'<br><br>"
    "5. <b>Interpretation</b> -- what did we learn about weather, not just about our model? "
    "If the city classifier confuses Dallas and Austin but never Dallas and LA, that tells "
    "us something real about climate geography.",
)

# ---------------------------------------------------------------------------
# Project selector
# ---------------------------------------------------------------------------
project = st.radio(
    "Select a Capstone Project:",
    [
        "Project 1: Best City Classifier",
        "Project 2: Temperature Forecast",
        "Project 3: Weather Pattern Discovery",
    ],
    key="capstone_project",
    horizontal=True,
)

st.divider()

# ============================================================================
# PROJECT 1: BEST CITY CLASSIFIER
# ============================================================================
if project == "Project 1: Best City Classifier":
    st.header("Project 1: Best City Classifier")
    st.markdown(
        "Here is the setup. I hand you a single weather reading: temperature = 31.2 degrees C, "
        "relative humidity = 78%, wind speed = 8.4 km/h, surface pressure = 1012.3 hPa. "
        "The city label has been stripped off. Which of our 6 cities -- Dallas, San Antonio, "
        "Houston, Austin, NYC, or Los Angeles -- produced this reading?"
    )
    st.markdown(
        "This is not just a fun game. It is a deep question about **climate fingerprints**. "
        "If a classifier can reliably identify a city from a single hour's weather, that means "
        "the cities have genuinely distinctive climate signatures. If it cannot, it means the "
        "cities' weather distributions overlap too much for any algorithm to distinguish. "
        "Either answer is interesting."
    )
    st.markdown(
        "My prediction before we start: the classifier will do well for LA (Mediterranean "
        "climate, low humidity, mild temperatures year-round) and NYC (colder winters, "
        "moderate humidity) but struggle to distinguish the Texas cities from each other, "
        "because Dallas, Austin, San Antonio, and Houston all share roughly similar "
        "subtropical patterns. Let us see if the data agrees."
    )

    # --- Step 1: Data Exploration ---
    st.subheader("Step 1: Data Exploration")

    st.markdown(
        "Before we train anything, we need to answer a basic question: **how different are "
        "these cities, really?** If all 6 cities have identical temperature distributions, "
        "no classifier in the world can tell them apart -- you cannot extract signal that "
        "does not exist. Conversely, if LA's humidity sits at 60% while Houston's sits at "
        "75%, that 15-percentage-point gap is free information the classifier can use."
    )

    # Per-city summary
    summary_rows = []
    for city in CITY_LIST:
        c = fdf[fdf["city"] == city]
        if len(c) == 0:
            continue
        summary_rows.append({
            "City": city,
            "N": f"{len(c):,}",
            "Temp Mean": f"{c['temperature_c'].mean():.1f}",
            "Temp Std": f"{c['temperature_c'].std():.1f}",
            "Humidity Mean": f"{c['relative_humidity_pct'].mean():.1f}",
            "Wind Mean": f"{c['wind_speed_kmh'].mean():.1f}",
            "Pressure Mean": f"{c['surface_pressure_hpa'].mean():.1f}",
        })

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.markdown(
        "Look at the table carefully. Where do you see the biggest differences? NYC's mean "
        "temperature is probably several degrees below the Texas cities. LA's humidity is "
        "likely lower than Houston's. Surface pressure varies with altitude -- cities at "
        "different elevations will have different baseline pressures. Each of these gaps is "
        "a 'handle' the classifier can grab."
    )

    # Feature distributions
    fig_dist = make_subplots(rows=2, cols=2, subplot_titles=[FEATURE_LABELS[f] for f in FEATURE_COLS])
    for i, feat in enumerate(FEATURE_COLS):
        row = i // 2 + 1
        col = i % 2 + 1
        for city in CITY_LIST:
            c_data = fdf[fdf["city"] == city][feat].dropna()
            if len(c_data) == 0:
                continue
            fig_dist.add_trace(
                go.Histogram(
                    x=c_data, name=city, opacity=0.5, nbinsx=50,
                    marker_color=CITY_COLORS.get(city, "#888"),
                    showlegend=(i == 0),
                ),
                row=row, col=col,
            )
    fig_dist.update_layout(template="plotly_white", height=600, barmode="overlay",
                           margin=dict(t=40, b=40))
    st.plotly_chart(fig_dist, use_container_width=True)

    insight_box(
        "The histograms tell the real story. Wherever you see city distributions that are "
        "clearly separated -- peaks sitting in different places, minimal overlap -- that feature "
        "is gold for the classifier. Wherever the histograms pile on top of each other, that "
        "feature is nearly useless for distinguishing cities. Pay special attention to surface "
        "pressure: if some cities sit at different altitudes, their pressure distributions "
        "might barely overlap at all, handing the classifier an almost-free feature. "
        "Temperature distributions, by contrast, probably overlap heavily between the Texas "
        "cities because they all experience similar summers and winters."
    )

    st.divider()

    # --- Step 2: Data Preparation ---
    st.subheader("Step 2: Data Preparation")

    st.markdown(
        "Now we set up the ML pipeline. Two decisions matter here: **which features to include** "
        "and **whether to standardize**. Standardization (subtracting the mean, dividing by "
        "standard deviation) matters because some algorithms -- logistic regression in particular "
        "-- are sensitive to feature scales. Temperature ranges from maybe -5 to 40 degrees C. "
        "Pressure ranges from 990 to 1030 hPa. Without standardization, the pressure feature "
        "dominates purely because its numbers are bigger, not because it is more informative."
    )

    col_prep1, col_prep2 = st.columns(2)
    with col_prep1:
        selected_features = st.multiselect(
            "Select features for the classifier",
            FEATURE_COLS,
            default=FEATURE_COLS,
            format_func=lambda c: FEATURE_LABELS.get(c, c),
            key="cap_features",
        )
    with col_prep2:
        test_size = st.slider("Test set size (%)", 10, 40, 20, key="cap_test_size")
        scale_data = st.checkbox("Standardise features", value=True, key="cap_scale")

    if not selected_features:
        st.warning("Select at least one feature.")
        st.stop()

    X_train, X_test, y_train, y_test, le, scaler = prepare_classification_data(
        fdf, selected_features, target="city", test_size=test_size / 100, scale=scale_data
    )

    st.markdown(f"**Training set**: {len(X_train):,} samples | **Test set**: {len(X_test):,} samples")
    st.markdown(f"**Features**: {', '.join([FEATURE_LABELS.get(f, f) for f in selected_features])}")
    st.markdown(f"**Classes**: {', '.join(le.classes_)}")

    st.markdown(
        "Try an experiment: deselect all features except temperature and see what happens "
        "to the accuracy below. Then try pressure alone. Then all four features together. "
        "The gaps in performance tell you which features carry the most distinctive city "
        "information."
    )

    st.divider()

    # --- Step 3: Model Comparison ---
    st.subheader("Step 3: Model Training and Comparison")

    st.markdown(
        "We are going to train three classifiers of increasing complexity. The point is not "
        "just to find the best one -- it is to see whether the added complexity buys us "
        "anything. If a logistic regression (essentially drawing straight-line boundaries "
        "between cities in feature space) achieves 85% accuracy, and a Random Forest "
        "(100 decision trees voting together) achieves 86%, that extra 1% probably is not "
        "worth the added complexity, training time, and interpretability cost."
    )
    st.markdown(
        "But if the gap is 65% vs 85%, that tells you something important: the decision "
        "boundaries between cities are genuinely nonlinear. A city might be identifiable not "
        "by any single feature but by the *combination* -- for example, Houston's signature "
        "might be high temperature AND high humidity AND moderate pressure, and only a model "
        "that can capture interactions will detect that pattern."
    )

    @st.cache_data(show_spinner="Training models...")
    def train_and_evaluate(_X_train, _X_test, _y_train, _y_test, _labels):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        }

        results = {}
        for name, model in models.items():
            model.fit(_X_train, _y_train)
            y_pred = model.predict(_X_test)
            metrics = classification_metrics(_y_test, y_pred, labels=_labels)
            results[name] = {
                "model": model,
                "y_pred": y_pred,
                "accuracy": metrics["accuracy"],
                "report": metrics["report"],
                "cm": metrics["confusion_matrix"],
            }
        return results

    results = train_and_evaluate(
        X_train.values, X_test.values, y_train, y_test, le.classes_.tolist()
    )

    # Accuracy comparison
    acc_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": [f"{r['accuracy']:.4f}" for r in results.values()],
    })
    st.dataframe(acc_df, use_container_width=True, hide_index=True)

    fig_acc = go.Figure(go.Bar(
        x=list(results.keys()),
        y=[r["accuracy"] for r in results.values()],
        marker_color=["#2A9D8F", "#F4A261", "#E63946"],
        text=[f"{r['accuracy']:.1%}" for r in results.values()],
        textposition="outside",
    ))
    fig_acc.update_layout(yaxis_title="Accuracy", yaxis_range=[0, 1.1])
    apply_common_layout(fig_acc, title="Model Accuracy Comparison", height=400)
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown(
        "Look at the actual numbers. Is the Random Forest meaningfully better, or are all "
        "three models clustered within a few percentage points? If they are all around 85-90%, "
        "that tells you the problem is 'easy' -- the city signatures are strong enough that "
        "even simple models can detect them. If there is a big spread, the extra complexity "
        "is capturing real nonlinear structure."
    )

    st.divider()

    # --- Step 4: Detailed Evaluation ---
    st.subheader("Step 4: Detailed Evaluation")

    st.markdown(
        "Overall accuracy is a useful headline number, but it hides the interesting details. "
        "Which cities does the model confuse? A model that is 88% accurate overall but gets "
        "LA right 99% of the time while confusing Dallas and Austin 40% of the time is telling "
        "you something important about climate similarity -- and about the limits of what "
        "weather features alone can distinguish."
    )

    best_model = max(results, key=lambda k: results[k]["accuracy"])
    st.markdown(f"**Best model: {best_model}** (accuracy: {results[best_model]['accuracy']:.1%})")

    # Confusion matrix
    fig_cm = plot_confusion_matrix(results[best_model]["cm"], le.classes_.tolist())
    fig_cm.update_layout(title=f"Confusion Matrix: {best_model}")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown(
        "The confusion matrix is the most informative graphic on this page. Read it row by "
        "row: each row is a true city, each column is what the model predicted. The diagonal "
        "shows correct classifications. Off-diagonal entries show confusions. **Which city "
        "pairs get confused most often?** I would bet they are geographically or climatically "
        "similar -- Dallas and Austin (both inland Texas), or Dallas and San Antonio. If the "
        "model ever confuses NYC with Houston, something is seriously wrong."
    )

    # Per-class metrics
    report = results[best_model]["report"]
    class_metrics = []
    for cls in le.classes_:
        if cls in report:
            class_metrics.append({
                "City": cls,
                "Precision": f"{report[cls]['precision']:.3f}",
                "Recall": f"{report[cls]['recall']:.3f}",
                "F1-Score": f"{report[cls]['f1-score']:.3f}",
                "Support": int(report[cls]["support"]),
            })
    st.dataframe(pd.DataFrame(class_metrics), use_container_width=True, hide_index=True)

    st.markdown(
        "**Precision** answers: 'When the model says Houston, how often is it actually "
        "Houston?' **Recall** answers: 'Of all the actual Houston readings, what fraction "
        "did the model catch?' If a city has high precision but low recall, the model is "
        "conservative -- it only calls something Houston when it is very sure, but it misses "
        "a lot of Houston readings. If precision is low but recall is high, the model is "
        "trigger-happy -- it labels too many things as Houston."
    )

    # Feature importance (Random Forest)
    if "Random Forest" in results:
        rf_model = results["Random Forest"]["model"]
        importances = rf_model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(f, f) for f in selected_features],
            "Importance": importances,
        }).sort_values("Importance", ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=imp_df["Importance"], y=imp_df["Feature"],
            orientation="h", marker_color="#7209B7",
        ))
        fig_imp.update_layout(xaxis_title="Importance")
        apply_common_layout(fig_imp, title="Random Forest Feature Importance", height=350)
        st.plotly_chart(fig_imp, use_container_width=True)

    insight_box(
        "The feature importance chart reveals which weather variables carry the most "
        "distinctive city information. If surface pressure dominates, that is likely because "
        "our cities sit at different elevations -- LA near sea level, others at various inland "
        "altitudes. If humidity is highly important, it is probably separating the humid "
        "subtropical cities (Houston) from drier ones (LA). Temperature is often less useful "
        "than you would expect, because all our cities experience similar seasonal ranges -- "
        "a 30 degrees C reading could come from any of them in summer. The model has learned "
        "what meteorologists already know: climate is about the full profile, not any single "
        "variable."
    )

    # Code
    code_example("""
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

features = ['temperature_c', 'relative_humidity_pct',
            'wind_speed_kmh', 'surface_pressure_hpa']
X = df[features].dropna()
y = df.loc[X.index, 'city']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_sc, y_train)
y_pred = rf.predict(X_test_sc)

print(classification_report(y_test, y_pred, target_names=le.classes_))
""")


# ============================================================================
# PROJECT 2: TEMPERATURE FORECAST
# ============================================================================
elif project == "Project 2: Temperature Forecast":
    st.header("Project 2: Temperature Forecast")
    st.markdown(
        "Here is the oldest question in meteorology, stripped down to its essentials: "
        "**what will the temperature be tomorrow?**"
    )
    st.markdown(
        "We have ~730 days of data per city (about 2 years of hourly readings, which we "
        "will aggregate to daily averages). We will use past temperatures, humidity, wind, "
        "pressure, and rolling statistics as inputs, and try to predict the next day's "
        "average temperature. This is a regression problem, and unlike the classification "
        "project, it comes with a built-in sanity check that is embarrassingly hard to beat."
    )
    st.markdown(
        "That sanity check is the **persistence forecast**: 'Tomorrow's temperature will "
        "be the same as today's.' This sounds absurdly lazy. But weather has enormous "
        "day-to-day autocorrelation -- if it is 25 degrees C today, there is a very good "
        "chance it will be somewhere between 23 and 27 degrees C tomorrow. The persistence "
        "forecast exploits this autocorrelation for free, without any model at all. If our "
        "fancy gradient boosting cannot beat 'same as today,' we have wasted our time."
    )

    # --- Step 1: Feature Engineering ---
    st.subheader("Step 1: Problem Setup and Feature Engineering")

    st.markdown(
        "**Select a city** below and watch how the feature engineering works. We create "
        "**lag features** (yesterday's temperature, the temperature 2 days ago, 3 days ago, "
        "a week ago), **rolling statistics** (the 7-day rolling mean and standard deviation), "
        "and **calendar features** (month, day of year) that encode seasonality. Each of "
        "these gives the model a different type of information: lags capture short-term "
        "persistence, rolling stats capture recent trends, and calendar features capture "
        "the long-term seasonal cycle."
    )

    fc_city = st.selectbox("City", CITY_LIST, key="cap_fc_city")

    city_data = fdf[fdf["city"] == fc_city].sort_values("datetime").copy()
    if len(city_data) < 200:
        st.warning("Not enough data for this city.")
        st.stop()

    # Create daily aggregates
    daily = city_data.groupby("date").agg({
        "temperature_c": "mean",
        "relative_humidity_pct": "mean",
        "wind_speed_kmh": "mean",
        "surface_pressure_hpa": "mean",
        "month": "first",
        "day_of_year": "first",
    }).reset_index()
    daily = daily.sort_values("date").reset_index(drop=True)

    # Create lag features
    for lag in [1, 2, 3, 7]:
        daily[f"temp_lag{lag}"] = daily["temperature_c"].shift(lag)
        if lag == 1:
            daily[f"humidity_lag{lag}"] = daily["relative_humidity_pct"].shift(lag)
            daily[f"wind_lag{lag}"] = daily["wind_speed_kmh"].shift(lag)
            daily[f"pressure_lag{lag}"] = daily["surface_pressure_hpa"].shift(lag)

    # Rolling features
    daily["temp_rolling7_mean"] = daily["temperature_c"].shift(1).rolling(7).mean()
    daily["temp_rolling7_std"] = daily["temperature_c"].shift(1).rolling(7).std()

    # Target: next day temperature
    daily["target_temp"] = daily["temperature_c"].shift(-1)

    # Drop rows with NaN
    daily = daily.dropna().reset_index(drop=True)

    st.markdown(f"**Dataset**: {len(daily)} daily observations for {fc_city}")

    feature_options = [c for c in daily.columns if c not in ["date", "target_temp", "temperature_c"]]
    selected_feats = st.multiselect(
        "Select features for forecasting",
        feature_options,
        default=["temp_lag1", "temp_lag2", "temp_lag7", "humidity_lag1",
                 "pressure_lag1", "temp_rolling7_mean", "month", "day_of_year"],
        key="cap_fc_feats",
    )

    if not selected_feats:
        st.warning("Select at least one feature.")
        st.stop()

    st.markdown(
        "The feature preview table below shows the first 10 rows. Notice how each row "
        "contains today's weather information (the lag features) and tomorrow's actual "
        "temperature (the target). This is the fundamental structure of a supervised "
        "forecasting problem: use the past to predict the future, but be very careful "
        "not to accidentally include future information in the features (that is called "
        "**data leakage**, and it is the most common way to get unrealistically good results)."
    )

    st.markdown("#### Feature Preview")
    st.dataframe(daily[["date"] + selected_feats + ["target_temp"]].head(10),
                 use_container_width=True, hide_index=True)

    st.divider()

    # --- Step 2: Baseline Models ---
    st.subheader("Step 2: Baseline Models")

    st.markdown(
        "This is the single most important step in the entire project, and the one most "
        "people skip. Before training any ML model, we establish two baselines that set the "
        "floor for 'acceptable performance':"
    )
    st.markdown(
        "- **Persistence**: 'Tomorrow will be the same as today.' If today is 25.3 degrees C, "
        "predict 25.3 degrees C for tomorrow. This exploits the massive autocorrelation in "
        "daily temperatures.\n"
        "- **Climatology**: 'Tomorrow will be the historical average for that day of year.' "
        "If day 180 (late June) has historically averaged 33.1 degrees C, predict 33.1. "
        "This exploits seasonality but ignores what is actually happening right now."
    )
    st.markdown(
        "These two baselines represent complementary strategies: persistence captures "
        "short-term continuity, climatology captures long-term patterns. A good ML model "
        "should capture *both* -- and ideally learn the interactions between them."
    )

    # Split chronologically
    split_idx = int(len(daily) * 0.8)
    train_df = daily.iloc[:split_idx]
    test_df = daily.iloc[split_idx:].copy()

    # Persistence baseline
    test_df["pred_persistence"] = test_df["temp_lag1"]

    # Climatology baseline
    clim = train_df.groupby("day_of_year")["temperature_c"].mean()
    test_df["pred_climatology"] = test_df["day_of_year"].map(clim)
    test_df["pred_climatology"] = test_df["pred_climatology"].fillna(train_df["temperature_c"].mean())

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    baselines = {}
    for name, pred_col in [("Persistence", "pred_persistence"), ("Climatology", "pred_climatology")]:
        valid = test_df[[pred_col, "target_temp"]].dropna()
        baselines[name] = {
            "RMSE": np.sqrt(mean_squared_error(valid["target_temp"], valid[pred_col])),
            "MAE": mean_absolute_error(valid["target_temp"], valid[pred_col]),
            "R2": r2_score(valid["target_temp"], valid[pred_col]),
        }

    st.markdown(
        f"**Persistence baseline RMSE**: {baselines['Persistence']['RMSE']:.2f} degrees C. "
        f"That is the number to beat. If your gradient boosting model with 100 trees and "
        f"8 carefully engineered features cannot do better than 'same as today,' the model "
        f"is adding complexity without value."
    )

    st.divider()

    # --- Step 3: ML Models ---
    st.subheader("Step 3: Machine Learning Models")

    st.markdown(
        "Now we train two actual models -- linear regression and gradient boosting -- and "
        "compare them against the baselines. **Linear regression** fits a weighted sum of "
        "the features: 'tomorrow = 0.8 * today + 0.1 * last_week + 0.05 * month + ...' "
        "This captures the basic idea that tomorrow depends mostly on today, with some "
        "seasonal adjustment. **Gradient boosting** builds 100 sequential decision trees, "
        "each one correcting the errors of the previous ones. It can capture nonlinear "
        "relationships that linear regression misses."
    )

    X_train = train_df[selected_feats]
    y_train = train_df["target_temp"]
    X_test = test_df[selected_feats]
    y_test = test_df["target_temp"]

    @st.cache_data(show_spinner="Training forecasting models...")
    def train_forecast_models(_X_train, _y_train, _X_test, _y_test):
        models = {
            "Linear Regression": LinearRegression(),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            ),
        }
        results = {}
        for name, model in models.items():
            model.fit(_X_train, _y_train)
            preds = model.predict(_X_test)
            results[name] = {
                "predictions": preds,
                "RMSE": np.sqrt(mean_squared_error(_y_test, preds)),
                "MAE": mean_absolute_error(_y_test, preds),
                "R2": r2_score(_y_test, preds),
                "model": model,
            }
        return results

    ml_results = train_forecast_models(X_train.values, y_train.values, X_test.values, y_test.values)

    # Combine all results
    all_results = {**baselines, **{k: {kk: vv for kk, vv in v.items() if kk != "predictions" and kk != "model"}
                                   for k, v in ml_results.items()}}

    results_df = pd.DataFrame(all_results).T
    results_df = results_df[["RMSE", "MAE", "R2"]].round(4)
    results_df.index.name = "Model"
    st.dataframe(results_df, use_container_width=True)

    st.markdown(
        "Read the results table carefully. Three things to check:\n"
        "1. **Do the ML models beat persistence?** If not, the features are not adding information "
        "beyond 'same as today.'\n"
        "2. **How big is the gap?** Going from RMSE = 3.5 to 3.2 degrees C is a 9% improvement. "
        "Going from 3.5 to 1.5 would be transformative. The size of the gap tells you how much "
        "predictable signal exists beyond simple autocorrelation.\n"
        "3. **Does gradient boosting beat linear regression?** If not, the temperature-feature "
        "relationship is approximately linear, and the extra complexity is wasted."
    )

    # Visualise
    fig_models = go.Figure()
    model_names = list(all_results.keys())
    rmse_vals = [all_results[m]["RMSE"] for m in model_names]
    fig_models.add_trace(go.Bar(
        x=model_names, y=rmse_vals,
        marker_color=["#ccc", "#ccc", "#2A9D8F", "#E63946"],
        text=[f"{v:.2f}" for v in rmse_vals],
        textposition="outside",
    ))
    fig_models.update_layout(yaxis_title="RMSE (deg C)")
    apply_common_layout(fig_models, title="Model Comparison: RMSE", height=400)
    st.plotly_chart(fig_models, use_container_width=True)

    st.divider()

    # --- Step 4: Forecast Visualisation ---
    st.subheader("Step 4: Forecast Visualisation")

    st.markdown(
        "Numbers in a table are useful, but the forecast plot below shows you what good "
        "and bad predictions actually look like. The solid line is actual temperature. The "
        "dotted red line is the ML model's forecast. The dashed gray line is the persistence "
        "baseline. Watch for places where the ML model clearly outperforms persistence -- "
        "those are moments when the model successfully predicted a temperature *change* that "
        "'same as today' could not."
    )

    best_ml = min(ml_results, key=lambda k: ml_results[k]["RMSE"])
    best_preds = ml_results[best_ml]["predictions"]

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=test_df["date"], y=test_df["target_temp"],
        mode="lines", name="Actual",
        line=dict(color="#264653", width=2),
    ))
    fig_fc.add_trace(go.Scatter(
        x=test_df["date"], y=best_preds,
        mode="lines", name=f"{best_ml} Forecast",
        line=dict(color="#E63946", width=1.5, dash="dot"),
    ))
    fig_fc.add_trace(go.Scatter(
        x=test_df["date"], y=test_df["pred_persistence"],
        mode="lines", name="Persistence",
        line=dict(color="#ccc", width=1, dash="dash"),
    ))
    fig_fc.update_layout(
        xaxis_title="Date", yaxis_title="Temperature (deg C)",
    )
    apply_common_layout(fig_fc, title=f"Temperature Forecast: {best_ml} vs Actual ({fc_city})", height=500)
    st.plotly_chart(fig_fc, use_container_width=True)

    # Residual analysis
    st.markdown(
        "Now the residual analysis -- this is where we check whether the model has any "
        "systematic blind spots. The residual is actual minus predicted. If the model is "
        "working well, the residual histogram should look like a bell curve centered at zero "
        "(no systematic bias, errors are random). The residuals-over-time plot should look "
        "like random scatter around zero. If you see a seasonal pattern in the residuals -- "
        "say, the model consistently underpredicts in summer -- that is the model leaving "
        "predictable signal on the table, and more features or a different model might help."
    )

    residuals = test_df["target_temp"].values - best_preds
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fig_resid = go.Figure(go.Histogram(
            x=residuals, nbinsx=50, marker_color="#7209B7", opacity=0.7,
        ))
        fig_resid.update_layout(xaxis_title="Residual (deg C)", yaxis_title="Count")
        apply_common_layout(fig_resid, title="Residual Distribution", height=350)
        st.plotly_chart(fig_resid, use_container_width=True)

    with col_r2:
        fig_resid_ts = go.Figure(go.Scatter(
            x=test_df["date"], y=residuals,
            mode="markers", marker=dict(color="#7209B7", size=3, opacity=0.5),
        ))
        fig_resid_ts.add_hline(y=0, line_dash="dash", line_color="black")
        fig_resid_ts.update_layout(xaxis_title="Date", yaxis_title="Residual (deg C)")
        apply_common_layout(fig_resid_ts, title="Residuals Over Time", height=350)
        st.plotly_chart(fig_resid_ts, use_container_width=True)

    insight_box(
        f"The {best_ml} model achieves RMSE of {ml_results[best_ml]['RMSE']:.2f} degrees C, "
        f"compared to the persistence baseline of {baselines['Persistence']['RMSE']:.2f} degrees C. "
        "That difference is the value added by ML -- the amount of predictable signal that "
        "exists in the lag features, rolling statistics, and seasonal patterns beyond simple "
        "autocorrelation. Notice how much of the heavy lifting is already done by the persistence "
        "baseline: weather is so autocorrelated that 'same as today' is an extremely competitive "
        "starting point. The ML model's job is to capture the *residual* structure -- the part "
        "that changes from day to day in a predictable way. That is a much harder task than "
        "predicting the level."
    )

    code_example("""
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Create daily averages and lag features
daily = city_df.resample('D', on='datetime').mean()
daily['temp_lag1'] = daily['temperature_c'].shift(1)
daily['temp_lag7'] = daily['temperature_c'].shift(7)
daily['rolling7'] = daily['temperature_c'].shift(1).rolling(7).mean()
daily['target'] = daily['temperature_c'].shift(-1)
daily = daily.dropna()

# Chronological split
split = int(len(daily) * 0.8)
train, test = daily.iloc[:split], daily.iloc[split:]

features = ['temp_lag1', 'temp_lag7', 'rolling7',
            'relative_humidity_pct', 'surface_pressure_hpa']

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(train[features], train['target'])
preds = model.predict(test[features])

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(test['target'], preds, squared=False)
print(f"RMSE: {rmse:.2f} deg C")
""")


# ============================================================================
# PROJECT 3: WEATHER PATTERN DISCOVERY
# ============================================================================
elif project == "Project 3: Weather Pattern Discovery":
    st.header("Project 3: Weather Pattern Discovery")
    st.markdown(
        "The first two projects were **supervised**: we had a known target (city label, "
        "tomorrow's temperature) and we measured success by how well we predicted it. This "
        "project is fundamentally different. We are going to hand the algorithm 15,000 "
        "weather readings -- each described by 4 numbers (temperature, humidity, wind, "
        "pressure) -- and ask: **what natural groupings exist in this data?**"
    )
    st.markdown(
        "We do not tell the algorithm how many groups to find, or what they should look "
        "like. We do not give it city labels or season labels. It just gets the raw numbers. "
        "The question is whether the structure it discovers corresponds to anything "
        "meteorologically meaningful."
    )
    st.markdown(
        "This is both more exciting and more dangerous than supervised learning. More "
        "exciting because we might discover weather regimes that we did not know to look "
        "for. More dangerous because there is no answer key -- the algorithm will always "
        "find clusters, even in pure random noise. Our job is to figure out whether the "
        "clusters are real or artifacts."
    )

    # --- Step 1: Data Preparation ---
    st.subheader("Step 1: Data Preparation for Clustering")

    st.markdown(
        "We standardize all 4 features before clustering. This is critical for K-Means "
        "because it measures distances in feature space. Without standardization, temperature "
        "(range: maybe -5 to 40 degrees C, span of 45) and pressure (range: 990 to 1030 hPa, "
        "span of 40) have similar numerical ranges, but humidity (0 to 100%) has a much "
        "larger numerical span, so it would dominate the distance calculations. "
        "Standardization gives each feature an equal vote."
    )

    cluster_data = fdf[FEATURE_COLS + ["city", "season", "month", "hour"]].dropna().copy()

    max_cluster_pts = 15000
    if len(cluster_data) > max_cluster_pts:
        cluster_data = cluster_data.sample(n=max_cluster_pts, random_state=42)

    st.markdown(f"**Working with {len(cluster_data):,} data points across {cluster_data['city'].nunique()} cities.**")

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_data[FEATURE_COLS])

    st.markdown(
        "The correlation matrix below shows how the 4 features relate to each other. "
        "Strong correlations (dark red or dark blue) mean two features carry similar "
        "information. If temperature and humidity are highly correlated (they often are -- "
        "warm air holds more moisture), the clusters will be driven partly by this shared "
        "variation rather than by independent information from each feature."
    )

    # Feature correlations
    corr = cluster_data[FEATURE_COLS].corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=[FEATURE_LABELS[f] for f in FEATURE_COLS],
        y=[FEATURE_LABELS[f] for f in FEATURE_COLS],
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
    ))
    apply_common_layout(fig_corr, title="Feature Correlation Matrix", height=400)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # --- Step 2: Finding Optimal K ---
    st.subheader("Step 2: Finding the Optimal Number of Clusters")

    st.markdown(
        "K-Means requires you to choose the number of clusters (k) in advance. This is "
        "the algorithm's biggest limitation: it cannot tell you how many natural groups "
        "exist. The **elbow method** helps: we run K-Means for k=2 through k=10 and plot "
        "the total within-cluster variance (inertia) for each. As k increases, inertia "
        "always decreases -- more clusters means tighter groups. But at some point, adding "
        "another cluster buys very little improvement. That 'bend' in the curve suggests "
        "the natural number of groups."
    )
    st.markdown(
        "For our weather data, I would expect the elbow around k=3 to k=5. Why? Because "
        "the dominant structure is probably seasonal (summer vs winter vs shoulder seasons) "
        "crossed with geographic (coastal vs inland), which gives maybe 4-6 natural regimes."
    )

    @st.cache_data(show_spinner="Computing elbow curve...")
    def compute_elbow(X, max_k=10):
        inertias = []
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append({"k": k, "inertia": km.inertia_})
        return pd.DataFrame(inertias)

    elbow_df = compute_elbow(X_scaled, max_k=10)

    fig_elbow = go.Figure(go.Scatter(
        x=elbow_df["k"], y=elbow_df["inertia"],
        mode="lines+markers",
        line=dict(color="#E63946", width=2),
        marker=dict(size=8),
    ))
    fig_elbow.update_layout(
        xaxis_title="Number of Clusters (k)", yaxis_title="Inertia",
    )
    apply_common_layout(fig_elbow, title="Elbow Method for Optimal K", height=400)
    st.plotly_chart(fig_elbow, use_container_width=True)

    st.markdown(
        "**Use the slider below** to choose k. Start with the elbow point, then try k-1 "
        "and k+1 to see how the clusters split or merge. If going from k=4 to k=5 creates "
        "a new cluster that looks meaningfully different from the others (say, a 'cold and "
        "windy' regime that was previously merged with 'cold and calm'), the extra cluster "
        "is probably real. If the new cluster just splits an existing one roughly in half "
        "with no clear physical interpretation, you have probably gone too far."
    )

    n_clusters = st.slider("Number of clusters", 2, 10, 4, key="cap_k")

    st.divider()

    # --- Step 3: Clustering ---
    st.subheader("Step 3: K-Means Clustering")

    @st.cache_data(show_spinner="Running K-Means...")
    def run_kmeans(X, k):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        centers = km.cluster_centers_
        return labels, centers

    labels, centers = run_kmeans(X_scaled, n_clusters)
    cluster_data["cluster"] = labels

    # PCA for visualisation
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    cluster_data["PC1"] = X_pca[:, 0]
    cluster_data["PC2"] = X_pca[:, 1]

    st.markdown(
        "The PCA plot below projects our 4-dimensional weather data onto 2 dimensions for "
        "visualization. Each point is one hourly weather reading, colored by its cluster "
        "assignment. The black X markers show cluster centers. Well-separated clusters in "
        "PCA space suggest genuinely distinct weather regimes. Overlapping clusters might "
        "still be meaningful -- they could be separated along dimensions that PCA does not "
        "emphasize -- but heavy overlap is a warning sign."
    )

    cluster_colors = ["#E63946", "#2A9D8F", "#F4A261", "#7209B7",
                      "#264653", "#FB8500", "#B5179E", "#0077B6",
                      "#00B4D8", "#90BE6D"]

    fig_pca = go.Figure()
    for k in range(n_clusters):
        mask = cluster_data["cluster"] == k
        fig_pca.add_trace(go.Scatter(
            x=cluster_data.loc[mask, "PC1"],
            y=cluster_data.loc[mask, "PC2"],
            mode="markers",
            marker=dict(color=cluster_colors[k % len(cluster_colors)], size=3, opacity=0.4),
            name=f"Cluster {k}",
        ))

    # Plot cluster centres
    centers_pca = pca.transform(centers)
    fig_pca.add_trace(go.Scatter(
        x=centers_pca[:, 0], y=centers_pca[:, 1],
        mode="markers",
        marker=dict(color="black", size=15, symbol="x", line=dict(width=2)),
        name="Centres",
    ))

    fig_pca.update_layout(
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
    )
    apply_common_layout(fig_pca, title="Clusters in PCA Space", height=500)
    st.plotly_chart(fig_pca, use_container_width=True)

    st.divider()

    # --- Step 4: Cluster Interpretation ---
    st.subheader("Step 4: Cluster Interpretation")

    st.markdown(
        "This is the hardest and most important part of any unsupervised analysis. K-Means "
        "will always partition data into k groups -- it will happily cluster pure random "
        "noise and report tidy results. The question is not 'did the algorithm find clusters?' "
        "(it always will) but 'do the clusters mean anything?'"
    )
    st.markdown(
        "Below we examine each cluster's weather profile: what is the average temperature, "
        "humidity, wind, and pressure? If one cluster is 'hot, humid, and calm' and another "
        "is 'cold, dry, and windy,' those sound like real weather regimes. If two clusters "
        "have nearly identical profiles, the algorithm is probably splitting noise."
    )

    # Cluster profiles
    profile_rows = []
    for k in range(n_clusters):
        mask = cluster_data["cluster"] == k
        cluster_subset = cluster_data[mask]
        row = {"Cluster": k, "Size": f"{mask.sum():,}"}
        for feat in FEATURE_COLS:
            row[FEATURE_LABELS[feat]] = f"{cluster_subset[feat].mean():.1f}"
        profile_rows.append(row)

    profile_df = pd.DataFrame(profile_rows)
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

    st.markdown(
        "The radar chart below normalizes each feature to a 0-1 scale and shows each "
        "cluster's profile as a polygon. Clusters with distinctive shapes -- one that is "
        "all 'hot and humid' versus one that is all 'cold and dry' -- represent genuinely "
        "different weather regimes. Similar shapes at different scales are less informative."
    )

    # Radar chart
    fig_radar = go.Figure()
    categories = [FEATURE_LABELS[f] for f in FEATURE_COLS]

    for k in range(n_clusters):
        mask = cluster_data["cluster"] == k
        values = []
        for feat in FEATURE_COLS:
            # Normalise to 0-1 for radar chart
            feat_min = cluster_data[feat].min()
            feat_max = cluster_data[feat].max()
            feat_range = feat_max - feat_min if feat_max > feat_min else 1
            values.append((cluster_data.loc[mask, feat].mean() - feat_min) / feat_range)

        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # close the polygon
            theta=categories + [categories[0]],
            fill="toself",
            name=f"Cluster {k}",
            line=dict(color=cluster_colors[k % len(cluster_colors)]),
            opacity=0.5,
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_white", height=500,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Cluster vs City
    st.markdown("#### Cluster Composition by City")
    st.markdown(
        "If the clusters are capturing real weather regimes, we would expect to see "
        "non-uniform distributions across cities. For instance, a 'hot and humid' cluster "
        "should be dominated by Houston and summer readings. A 'cold and dry' cluster "
        "might be mostly NYC winter data. If every cluster has exactly 1/6 of each city, "
        "the clusters are not capturing geographic climate differences."
    )

    city_cluster = pd.crosstab(cluster_data["city"], cluster_data["cluster"], normalize="index")
    city_cluster.columns = [f"Cluster {c}" for c in city_cluster.columns]

    fig_comp = go.Figure()
    for col in city_cluster.columns:
        fig_comp.add_trace(go.Bar(
            x=city_cluster.index, y=city_cluster[col],
            name=col,
        ))
    fig_comp.update_layout(barmode="stack", xaxis_title="City", yaxis_title="Proportion")
    apply_common_layout(fig_comp, title="Cluster Composition by City", height=400)
    st.plotly_chart(fig_comp, use_container_width=True)

    # Cluster vs Season
    st.markdown("#### Cluster Composition by Season")
    st.markdown(
        "Similarly, the seasonal breakdown reveals whether the clusters are picking up "
        "temporal structure. A cluster that appears mostly in summer and another mostly in "
        "winter is rediscovering seasonality from raw weather numbers -- without being told "
        "what season any reading came from."
    )

    season_cluster = pd.crosstab(cluster_data["season"], cluster_data["cluster"], normalize="index")
    season_cluster.columns = [f"Cluster {c}" for c in season_cluster.columns]
    season_cluster = season_cluster.reindex(["Winter", "Spring", "Summer", "Fall"])

    fig_seas = go.Figure()
    for col in season_cluster.columns:
        fig_seas.add_trace(go.Bar(
            x=season_cluster.index, y=season_cluster[col],
            name=col,
        ))
    fig_seas.update_layout(barmode="stack", xaxis_title="Season", yaxis_title="Proportion")
    apply_common_layout(fig_seas, title="Cluster Composition by Season", height=400)
    st.plotly_chart(fig_seas, use_container_width=True)

    # Name the clusters
    st.markdown("#### Cluster Labels (Based on Feature Profiles)")
    st.markdown(
        "Below we auto-generate descriptive labels based on each cluster's average "
        "weather values. These labels are heuristic -- based on whether the cluster's "
        "mean falls in the top 30%, middle 40%, or bottom 30% of each feature's "
        "distribution. They give a rough sense of what each cluster 'is,' but always "
        "check the actual numbers in the profile table above."
    )

    for k in range(n_clusters):
        mask = cluster_data["cluster"] == k
        cs = cluster_data[mask]
        temp = cs["temperature_c"].mean()
        hum = cs["relative_humidity_pct"].mean()
        wind = cs["wind_speed_kmh"].mean()
        pres = cs["surface_pressure_hpa"].mean()

        temp_label = "Hot" if temp > cluster_data["temperature_c"].quantile(0.7) else \
                     "Cold" if temp < cluster_data["temperature_c"].quantile(0.3) else "Mild"
        hum_label = "Humid" if hum > cluster_data["relative_humidity_pct"].quantile(0.7) else \
                    "Dry" if hum < cluster_data["relative_humidity_pct"].quantile(0.3) else "Moderate"
        wind_label = "Windy" if wind > cluster_data["wind_speed_kmh"].quantile(0.7) else "Calm"

        label = f"**Cluster {k}**: {temp_label}, {hum_label}, {wind_label} "
        label += f"(Temp={temp:.1f}C, Hum={hum:.0f}%, Wind={wind:.1f}km/h)"
        top_city = cs["city"].mode().iloc[0] if len(cs) > 0 else "N/A"
        top_season = cs["season"].mode().iloc[0] if len(cs) > 0 else "N/A"
        label += f" -- Most common in {top_city}, {top_season}"
        st.markdown(label)

    insight_box(
        "Here is what I find genuinely remarkable about this analysis. The clustering "
        "algorithm received nothing but 4 numbers per data point -- no city labels, no "
        "season labels, no geographic information. And yet the clusters it discovers "
        "independently correspond to weather regimes that meteorologists would recognize: "
        "hot-humid summer, cold-dry winter, mild transitional seasons, coastal moderation. "
        "The 'Cluster Composition by City' chart is the smoking gun: if Cluster 0 is "
        "dominated by Houston summer readings and Cluster 2 by NYC winter readings, the "
        "algorithm has rediscovered climate geography from raw thermometer data. This is "
        "either a powerful validation of unsupervised learning or evidence that weather "
        "patterns are so strong that any reasonable algorithm would find them. Either way, "
        "the clusters are not arbitrary -- they reflect physical reality."
    )

    code_example("""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

features = ['temperature_c', 'relative_humidity_pct',
            'wind_speed_kmh', 'surface_pressure_hpa']
X = df[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k
inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Fit with chosen k
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualise with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Interpret clusters
for k in range(4):
    mask = df['cluster'] == k
    print(f"Cluster {k}: n={mask.sum()}")
    for f in features:
        print(f"  {f}: {df.loc[mask, f].mean():.1f}")
""")

st.divider()

# ---------------------------------------------------------------------------
# Final Takeaways (shared)
# ---------------------------------------------------------------------------
takeaways([
    "The data science pipeline -- explore, prepare, model, evaluate, interpret -- is not "
    "a linear sequence you walk through once. It is a loop. The evaluation step reveals "
    "problems (model confuses Dallas and Austin), which sends you back to preparation "
    "(add elevation as a feature?) or exploration (what makes Dallas and Austin so similar?). "
    "Real projects go around this loop 5-10 times.",
    "Always establish baselines before building complex models. The persistence forecast "
    "('tomorrow equals today') is the temperature forecasting equivalent of 'first, do no "
    "harm.' If your gradient boosting with 100 trees and 8 features cannot beat this trivial "
    "baseline, it is adding complexity without value. In our dataset, the persistence RMSE "
    "is typically around 2-4 degrees C depending on the city -- that is the bar.",
    "Feature engineering -- lag features, rolling statistics, calendar encodings -- is where "
    "the real performance gains live. The difference between a good model and a great model "
    "is rarely the algorithm; it is what you feed it. In the temperature forecast project, "
    "the 7-day rolling mean alone captures most of the seasonal structure.",
    "Unsupervised methods always find structure -- that is literally what they are designed "
    "to do. The hard part is determining whether the structure is real. For the clustering "
    "project, we validated clusters against city labels and season labels that the algorithm "
    "never saw. If the clusters had not aligned with these external variables, we would have "
    "less confidence they were real. Always validate against domain knowledge.",
    "The confusion matrix in the classification project and the residual analysis in the "
    "forecasting project are not afterthoughts -- they are the most informative parts of "
    "the entire analysis. They tell you not just how well the model works, but *where* "
    "and *why* it fails. A model that confuses Dallas and Austin is telling you something "
    "real about climate similarity. A model with seasonal residual patterns is telling you "
    "it needs seasonal features.",
    "Every tool we used in these projects -- Random Forests, gradient boosting, K-Means, "
    "PCA, cross-tabulation, residual analysis -- was introduced in isolation in earlier "
    "chapters. The skill of data science is not knowing how to use each tool; it is knowing "
    "when to reach for which tool, and how to combine them into a coherent analysis that "
    "answers a real question about the world.",
])

st.divider()

st.markdown("""
### You Made It

Sixty-two chapters. We started with histograms and means -- literally just counting things
and computing averages. Now we are building city classifiers from climate fingerprints,
forecasting temperatures with gradient boosting, and discovering hidden weather regimes
with unsupervised learning. That is a genuinely large distance to have covered.

Here is an incomplete inventory of what you now know:

- **Statistical foundations**: distributions, confidence intervals, hypothesis testing --
  the grammar of making defensible claims about data. You know that a p-value of 0.03
  does not mean there is a 3% chance the null is true, and you know why that distinction
  matters.
- **Visualization**: not just making charts, but making charts that reveal structure
  rather than hide it. You have seen how a well-chosen histogram can tell you more than
  a table of summary statistics, and how a poorly chosen visualization can actively mislead.
- **Machine learning**: classification (Random Forests, logistic regression), regression
  (linear, gradient boosting), dimensionality reduction (PCA), clustering (K-Means) --
  the supervised and unsupervised toolkit. You understand the bias-variance tradeoff and
  why adding model complexity does not always help.
- **Time series**: forecasting, seasonality, autocorrelation, lag features -- because
  weather data has a time axis and ignoring it is not just a rookie mistake, it is a
  fundamental violation of the data's structure.
- **Deep learning**: neural networks and autoencoders -- for problems where the
  relationships are complex enough to justify the complexity of the solution, and for
  anomaly detection through reconstruction error.
- **Bayesian methods**: Bayes' theorem, prior and posterior distributions, probabilistic
  reasoning -- for when you want to reason about uncertainty honestly rather than pretending
  a point estimate tells the whole story.
- **Anomaly detection**: Z-scores, Isolation Forest, autoencoders -- three fundamentally
  different approaches to finding the weird stuff, which is often the most interesting stuff.
- **Causal inference**: correlation versus causation, confounders, Simpson's paradox,
  natural experiments, difference-in-differences -- for when you want to understand *why*
  things happen, not just predict that they will.

All of it illustrated with real weather data from 6 US cities -- 105,264 hourly readings
spanning 2 years. Weather turned out to be a surprisingly rich domain for data science:
it is messy, seasonal, spatial, multivariate, autocorrelated, and full of both obvious
patterns (summer is hot) and subtle structure (pressure changes precede wind changes).
Much like data science itself.

Now go build something.
""")

navigation(
    prev_label="Ch 61: Natural Experiments",
    prev_page="61_Natural_Experiments.py",
)
