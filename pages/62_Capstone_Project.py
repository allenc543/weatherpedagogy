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
st.set_page_config(page_title="Ch 62: Capstone Project", layout="wide")
df = load_data()
fdf = sidebar_filters(df)

chapter_header(62, "Capstone Project", part="XV")

concept_box(
    "End-to-End Data Science Workflow",
    "We've spent 61 chapters learning every tool in the data science toolkit. Now it's "
    "time to actually use them all at once, which is roughly the difference between learning "
    "to juggle individual balls and being asked to juggle while riding a unicycle.<br><br>"
    "Each project below walks through the <b>complete data science pipeline</b>:<br>"
    "1. <b>Problem definition</b> -- what are we actually trying to figure out, and why should anyone care?<br>"
    "2. <b>Data exploration</b> -- staring at the data until it confesses its secrets<br>"
    "3. <b>Model building</b> -- the part everyone thinks is the whole job<br>"
    "4. <b>Evaluation</b> -- the part that actually IS the whole job<br>"
    "5. <b>Conclusions</b> -- what we learned and what we're still confused about",
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
        "**Goal**: Here's a fun question -- if I hand you a weather reading (temperature, "
        "humidity, wind speed, pressure) with the city label stripped off, can you figure out "
        "which city it came from? This is secretly a deep question about how distinctive "
        "different climates really are. Let's build a full ML pipeline to find out."
    )

    # --- Step 1: Data Exploration ---
    st.subheader("Step 1: Data Exploration")

    st.markdown(
        "Before we throw algorithms at the problem, we should do the thing that separates "
        "good data scientists from people who just import sklearn and pray: actually look "
        "at our data. How do these cities differ across weather features? Are there obvious "
        "signatures, or is this going to be hard?"
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
        "This is the key intuition: wherever you see city distributions that barely overlap, "
        "you've found a feature the classifier can exploit. Wherever distributions pile on "
        "top of each other like a rugby scrum, that feature is basically useless for telling "
        "cities apart. The model will figure this out too, but you should figure it out first -- "
        "because if the model disagrees with your eyes, someone made a mistake."
    )

    st.divider()

    # --- Step 2: Data Preparation ---
    st.subheader("Step 2: Data Preparation")

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

    st.divider()

    # --- Step 3: Model Comparison ---
    st.subheader("Step 3: Model Training and Comparison")

    st.markdown(
        "Now for the horse race. We'll train three classifiers of increasing sophistication "
        "and see which one earns its complexity budget. If the simple model does just as well "
        "as the fancy one, Occam's Razor says we should prefer it -- and also that we wasted "
        "a bunch of compute on the fancy one, but at least we learned something."
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

    st.divider()

    # --- Step 4: Detailed Evaluation ---
    st.subheader("Step 4: Detailed Evaluation")

    best_model = max(results, key=lambda k: results[k]["accuracy"])
    st.markdown(f"**Best model: {best_model}** (accuracy: {results[best_model]['accuracy']:.1%})")

    # Confusion matrix
    fig_cm = plot_confusion_matrix(results[best_model]["cm"], le.classes_.tolist())
    fig_cm.update_layout(title=f"Confusion Matrix: {best_model}")
    st.plotly_chart(fig_cm, use_container_width=True)

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
        "The Random Forest typically wins this contest, and the reason is genuinely interesting: "
        "it can capture non-linear interactions between features that logistic regression treats "
        "as invisible. A city might have mild temperatures AND high humidity AND low pressure -- "
        "it's the combination that's distinctive, not any single reading. Meanwhile, the confusion "
        "matrix tells you which city pairs the model keeps mixing up, and I'd bet a nontrivial "
        "amount that they're geographically close. If your model confuses Phoenix and Miami, "
        "something has gone very wrong."
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
        "**Goal**: The oldest question in weather: what will the temperature be tomorrow? "
        "This sounds simple until you try it. We'll start with the dumbest possible approach "
        "(\"tomorrow will be like today\"), then see how much improvement we can buy with actual "
        "machine learning. The gap between the dumb approach and the smart one is, in some deep "
        "sense, a measure of how predictable weather actually is."
    )

    # --- Step 1: Feature Engineering ---
    st.subheader("Step 1: Problem Setup and Feature Engineering")

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

    st.markdown("#### Feature Preview")
    st.dataframe(daily[["date"] + selected_feats + ["target_temp"]].head(10),
                 use_container_width=True, hide_index=True)

    st.divider()

    # --- Step 2: Baseline Models ---
    st.subheader("Step 2: Baseline Models")

    st.markdown(
        "Here is the most important step in any forecasting project, and the one most people "
        "skip: establishing baselines. If you can't beat a trivially simple method, your "
        "fancy model is adding complexity without value -- which is, I would argue, the "
        "central sin of applied machine learning.\n\n"
        "- **Persistence**: \"Tomorrow will be the same as today.\" This is the laziest "
        "possible forecast, and it's shockingly hard to beat for short horizons.\n"
        "- **Climatology**: \"Tomorrow will be the historical average for this day of year.\" "
        "This captures seasonality but ignores what's actually happening right now."
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

    st.divider()

    # --- Step 3: ML Models ---
    st.subheader("Step 3: Machine Learning Models")

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
        f"The {best_ml} model achieves RMSE of {ml_results[best_ml]['RMSE']:.2f} deg C, "
        f"beating the persistence baseline ({baselines['Persistence']['RMSE']:.2f} deg C). "
        "That gap is the value added by actually doing data science instead of just saying "
        "\"eh, same as today.\" But notice how much of the work is already done by the naive "
        "approach -- weather has enormous autocorrelation, so \"tomorrow equals today\" gets "
        "you surprisingly far. The residual plots should look like random noise centered at "
        "zero; if you see patterns, the model is leaving predictable signal on the table."
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
        "**Goal**: This project asks a fundamentally different kind of question. Instead of "
        "\"given inputs X, predict label Y,\" we're asking: \"what natural groupings exist in "
        "this data that we didn't already know about?\" This is unsupervised learning, and it's "
        "both more exciting and more treacherous than the supervised kind -- more exciting "
        "because you might discover something genuinely new, more treacherous because there's "
        "no answer key to check against. Do the clusters correspond to real meteorological "
        "phenomena, or did we just find noise that looks organized? Let's find out."
    )

    # --- Step 1: Data Preparation ---
    st.subheader("Step 1: Data Preparation for Clustering")

    cluster_data = fdf[FEATURE_COLS + ["city", "season", "month", "hour"]].dropna().copy()

    max_cluster_pts = 15000
    if len(cluster_data) > max_cluster_pts:
        cluster_data = cluster_data.sample(n=max_cluster_pts, random_state=42)

    st.markdown(f"**Working with {len(cluster_data):,} data points across {cluster_data['city'].nunique()} cities.**")

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_data[FEATURE_COLS])

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
        "Now the hard part. K-Means will always give you clusters -- it'll happily partition "
        "random noise into tidy groups and feel very proud of itself. The question is whether "
        "these clusters mean anything. Let's look at each cluster's weather profile and see if "
        "they correspond to patterns a meteorologist would recognize."
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
        "Here's what I find genuinely delightful about this: the clusters independently "
        "rediscover weather regimes that meteorologists already know about. Hot-dry clusters "
        "show up in summer inland cities. Mild-humid clusters dominate coastal areas and "
        "transitional seasons. The algorithm has no idea what \"summer\" or \"coastal\" means -- "
        "it's just looking at numbers -- but it converges on the same categories that humans "
        "built from decades of domain expertise. This is either a reassuring validation of "
        "unsupervised learning, or evidence that weather patterns are so strong they'd be "
        "hard to miss. Probably both."
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
    "A real data science project is not a sequence of techniques -- it's a conversation with "
    "your data where you keep asking questions and revising your understanding. The pipeline "
    "(explore, preprocess, model, evaluate, interpret) is less a rigid procedure than a loop "
    "you go around multiple times.",
    "Always, always, always establish baselines before building complex models. If you can't "
    "beat 'tomorrow equals today,' your fancy gradient boosting is adding complexity for "
    "nothing. This is the data science equivalent of 'first, do no harm.'",
    "Feature engineering -- lag features, rolling statistics, domain-specific transformations "
    "-- is often where the real performance gains live. The choice of model matters less than "
    "the quality of what you feed it.",
    "Unsupervised methods are powerful precisely because they find structure you didn't "
    "tell them to look for. But this means they can also find structure that isn't real. "
    "Always validate clusters against domain knowledge.",
    "Speaking of domain knowledge: meteorology (or whatever your application domain is) "
    "is not optional. It's what lets you tell the difference between a real pattern and a "
    "statistical mirage.",
    "Comparing multiple models and methods isn't just good practice -- it's how you develop "
    "calibrated intuitions about what kinds of problems benefit from what kinds of approaches.",
])

st.divider()

st.markdown("""
### You Made It

We started 62 chapters ago with histograms and means, and now we're building end-to-end ML
pipelines, forecasting temperatures, and discovering hidden weather regimes with unsupervised
learning. That's a genuinely large distance to have covered.

Here's what you've picked up along the way:

- **Statistical foundations**: distributions, inference, hypothesis testing -- the grammar
  of making claims about data
- **Visualization**: not just making charts, but making charts that actually reveal what's
  going on rather than what you hoped was going on
- **Machine learning**: classification, regression, clustering, dimensionality reduction --
  the full supervised and unsupervised toolkit
- **Time series**: forecasting, seasonality, trends -- because most real-world data has a
  time axis and ignoring it is a rookie mistake
- **Deep learning**: neural networks, autoencoders -- for when the problem is complex enough
  to justify the complexity of the solution
- **Bayesian methods**: Bayes' theorem, inference, probabilistic programming -- for when you
  want to reason about uncertainty honestly instead of sweeping it under a p-value
- **Anomaly detection**: statistical, tree-based, and neural approaches -- for finding the
  weird stuff, which is often the interesting stuff
- **Causal inference**: correlation vs causation, natural experiments -- for when you want
  to actually understand why things happen, not just predict that they will

All of it illustrated with real weather data from 6 US cities, which turns out to be a
surprisingly rich playground for data science concepts. Weather is messy, seasonal, spatial,
multivariate, and endlessly interesting -- much like the field itself.

Now go build something.
""")

navigation(
    prev_label="Ch 61: Natural Experiments",
    prev_page="61_Natural_Experiments.py",
)
