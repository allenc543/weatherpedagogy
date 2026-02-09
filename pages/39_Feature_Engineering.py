"""Chapter 39 -- Feature Engineering for Weather Data."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(39, "Feature Engineering Basics", part="IX")

st.markdown(
    "Feature engineering is the part of data science that is closest to actual "
    "wizardry. You take your raw data columns -- timestamps, temperatures, humidity "
    "readings -- and transform them into inputs that make a model's job dramatically "
    "easier. It is the difference between handing someone a pile of lumber and handing "
    "them a finished bookshelf. A good feature encodes domain knowledge in a form "
    "that algorithms can actually use. This chapter walks through the most important "
    "techniques, and by the end, you will see a Random Forest's MAE drop like a stone "
    "as we add engineered features one category at a time."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

concept_box(
    "What Is Feature Engineering?",
    "Feature engineering is the art and science of creating new input variables "
    "from existing data. It is where domain expertise meets machine learning. "
    "Common categories:<br>"
    "- <b>Datetime features</b>: hour, day of week, month, season -- because "
    "algorithms do not know what a calendar is<br>"
    "- <b>Lag features</b>: previous time step values -- 'what was the "
    "temperature 24 hours ago?'<br>"
    "- <b>Rolling statistics</b>: moving averages, rolling std dev -- smoothed "
    "versions of the signal<br>"
    "- <b>Domain features</b>: dew point, heat index, wind chill -- physically "
    "meaningful derived quantities"
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.subheader("Feature Engineering Settings")
city = st.sidebar.selectbox("City", CITY_LIST, key="fe_city")

city_df = fdf[fdf["city"] == city].sort_values("datetime").copy()

# ── Section 1: Datetime Features ─────────────────────────────────────────────
st.header("1. Datetime Features")

st.markdown(
    "The timestamp is hiding valuable information in plain sight. Extracting "
    "the hour, month, or day-of-year immediately gives the model information "
    "about cyclical patterns. But there is a subtlety: you cannot just feed "
    "the raw integer. If you tell a model 'hour = 23' and 'hour = 0,' it thinks "
    "those are maximally different (23 units apart!), when in reality they are "
    "one hour apart. The solution is cyclical encoding with sin/cos pairs."
)

use_hour = st.checkbox("Add `hour` feature", value=True, key="fe_hour")
use_month = st.checkbox("Add `month` feature", value=True, key="fe_month")
use_dayofyear = st.checkbox("Add `day_of_year` feature", value=False, key="fe_doy")

feat_df = city_df[["datetime", "temperature_c", "relative_humidity_pct",
                     "wind_speed_kmh", "surface_pressure_hpa", "hour", "month",
                     "day_of_year"]].copy()

datetime_feats = []
if use_hour:
    # Cyclical encoding
    feat_df["hour_sin"] = np.sin(2 * np.pi * feat_df["hour"] / 24)
    feat_df["hour_cos"] = np.cos(2 * np.pi * feat_df["hour"] / 24)
    datetime_feats.extend(["hour_sin", "hour_cos"])
if use_month:
    feat_df["month_sin"] = np.sin(2 * np.pi * feat_df["month"] / 12)
    feat_df["month_cos"] = np.cos(2 * np.pi * feat_df["month"] / 12)
    datetime_feats.extend(["month_sin", "month_cos"])
if use_dayofyear:
    feat_df["doy_sin"] = np.sin(2 * np.pi * feat_df["day_of_year"] / 365.25)
    feat_df["doy_cos"] = np.cos(2 * np.pi * feat_df["day_of_year"] / 365.25)
    datetime_feats.extend(["doy_sin", "doy_cos"])

if datetime_feats:
    st.markdown("**Preview of cyclical datetime features:**")
    st.dataframe(feat_df[["datetime"] + datetime_feats].head(10), use_container_width=True)

    # Visualize cyclical encoding
    if use_hour:
        fig_cyc = px.scatter(
            feat_df.head(200), x="hour_sin", y="hour_cos",
            color=feat_df["hour"].head(200),
            color_continuous_scale="twilight",
            title="Cyclical Encoding of Hour",
            labels={"color": "Hour"},
        )
        apply_common_layout(fig_cyc, height=400)
        st.plotly_chart(fig_cyc, use_container_width=True)

    insight_box(
        "The sin/cos encoding maps hours onto a circle, so hour 23 and hour 0 "
        "end up as neighbors (because they *are* neighbors). This dramatically "
        "helps distance-based models like KNN, but even tree-based models benefit "
        "because the cyclical structure is now explicit rather than something the "
        "tree has to painfully reconstruct through many splits."
    )

formula_box(
    "Cyclical Encoding",
    r"\text{hour\_sin} = \sin\!\left(\frac{2\pi \cdot h}{24}\right), \quad "
    r"\text{hour\_cos} = \cos\!\left(\frac{2\pi \cdot h}{24}\right)",
    "The (sin, cos) pair traces a circle. Two hours that are close in real life "
    "are close in this encoding, regardless of where the 'midnight boundary' falls."
)

# ── Section 2: Lag Features ──────────────────────────────────────────────────
st.header("2. Lag Features")

st.markdown(
    "Lag features are beautifully simple: use past values as predictors. "
    "For hourly temperature forecasting, a lag of 24 hours is especially "
    "powerful, because the diurnal cycle means that 2 PM today looks a lot "
    "like 2 PM yesterday. It is the same principle that makes the seasonal "
    "naive forecast work, except we are giving this information to a model "
    "that can combine it with other features."
)

use_lag1 = st.checkbox("Add `temp_lag_1h` (1-hour lag)", value=True, key="fe_lag1")
use_lag24 = st.checkbox("Add `temp_lag_24h` (24-hour lag)", value=True, key="fe_lag24")

lag_feats = []
if use_lag1:
    feat_df["temp_lag_1h"] = feat_df["temperature_c"].shift(1)
    lag_feats.append("temp_lag_1h")
if use_lag24:
    feat_df["temp_lag_24h"] = feat_df["temperature_c"].shift(24)
    lag_feats.append("temp_lag_24h")

if lag_feats:
    # Show correlation of lags with target
    lag_corrs = feat_df[["temperature_c"] + lag_feats].corr()["temperature_c"].drop("temperature_c")
    st.markdown("**Correlation of lag features with current temperature:**")
    for feat, corr in lag_corrs.items():
        st.markdown(f"- `{feat}`: r = **{corr:.4f}**")

    if use_lag24:
        fig_lag = px.scatter(
            feat_df.dropna().sample(n=min(3000, len(feat_df.dropna())), random_state=42),
            x="temp_lag_24h", y="temperature_c",
            opacity=0.2, title="Temperature vs 24h Lag",
            labels={"temp_lag_24h": "Temperature 24h Ago (deg C)",
                    "temperature_c": "Current Temperature (deg C)"},
            color_discrete_sequence=["#2E86C1"],
        )
        apply_common_layout(fig_lag, height=400)
        st.plotly_chart(fig_lag, use_container_width=True)

    insight_box(
        "The 24-hour lag has a remarkably high correlation with current "
        "temperature. This is the diurnal cycle doing its thing: today at 2 PM "
        "is a lot like yesterday at 2 PM, because the sun follows essentially "
        "the same path. This single feature is often the most valuable input "
        "for temperature forecasting."
    )

# ── Section 3: Rolling Statistics ────────────────────────────────────────────
st.header("3. Rolling Statistics")

st.markdown(
    "Rolling windows are the data science equivalent of squinting at a painting "
    "from across the room -- they blur out the fine detail to reveal the broader "
    "pattern. A 24-hour rolling mean gives you the average temperature over the "
    "past day, which captures whether we are in a generally warm or cold period "
    "without getting distracted by the hour-to-hour jitter."
)

use_roll24 = st.checkbox("Add `temp_rolling_24h_mean`", value=True, key="fe_roll24")
use_roll24_std = st.checkbox("Add `temp_rolling_24h_std`", value=False, key="fe_roll24s")
use_roll168 = st.checkbox("Add `temp_rolling_7d_mean`", value=False, key="fe_roll168")

roll_feats = []
if use_roll24:
    feat_df["temp_rolling_24h_mean"] = feat_df["temperature_c"].rolling(24).mean()
    roll_feats.append("temp_rolling_24h_mean")
if use_roll24_std:
    feat_df["temp_rolling_24h_std"] = feat_df["temperature_c"].rolling(24).std()
    roll_feats.append("temp_rolling_24h_std")
if use_roll168:
    feat_df["temp_rolling_7d_mean"] = feat_df["temperature_c"].rolling(168).mean()
    roll_feats.append("temp_rolling_7d_mean")

if roll_feats:
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=feat_df["datetime"].iloc[:500],
        y=feat_df["temperature_c"].iloc[:500],
        mode="lines", name="Actual", line=dict(color="#264653", width=1),
        opacity=0.5,
    ))
    colors = ["#E63946", "#2A9D8F", "#7209B7"]
    for i, rf in enumerate(roll_feats):
        fig_roll.add_trace(go.Scatter(
            x=feat_df["datetime"].iloc[:500],
            y=feat_df[rf].iloc[:500],
            mode="lines", name=rf, line=dict(color=colors[i % len(colors)], width=2),
        ))
    apply_common_layout(fig_roll, "Rolling Features vs Actual Temperature", 400)
    st.plotly_chart(fig_roll, use_container_width=True)

# ── Section 4: Domain-Specific Features ──────────────────────────────────────
st.header("4. Domain-Specific Features")

st.markdown(
    "This is where domain knowledge earns its keep. Meteorologists have spent "
    "decades figuring out physically meaningful derived quantities like **dew "
    "point** (the temperature at which air becomes saturated) and **apparent "
    "temperature** (what it 'feels like' accounting for humidity and wind). "
    "These features work well because they capture real physical relationships "
    "that raw columns cannot express on their own."
)

use_dewpoint = st.checkbox("Add `dew_point_estimate`", value=True, key="fe_dew")
use_temp_range = st.checkbox("Add `daily_temp_range` (max - min per day)", value=False, key="fe_range")

domain_feats = []
if use_dewpoint:
    # Magnus approximation for dew point
    feat_df["dew_point_estimate"] = (
        feat_df["temperature_c"]
        - ((100 - feat_df["relative_humidity_pct"]) / 5)
    )
    domain_feats.append("dew_point_estimate")

if use_temp_range:
    daily_range = city_df.groupby("date")["temperature_c"].agg(["min", "max"])
    daily_range["daily_temp_range"] = daily_range["max"] - daily_range["min"]
    city_df_tmp = city_df.merge(
        daily_range[["daily_temp_range"]].reset_index(),
        on="date", how="left",
    )
    feat_df["daily_temp_range"] = city_df_tmp["daily_temp_range"].values
    domain_feats.append("daily_temp_range")

if use_dewpoint:
    formula_box(
        "Dew Point Approximation",
        r"T_{dew} \approx T - \frac{100 - RH}{5}",
        "This is a quick-and-dirty approximation that works surprisingly well "
        "for moderate humidity levels. The real formula involves logarithms, but "
        "this gets you within a degree or two in most conditions."
    )

    fig_dew = px.scatter(
        feat_df.dropna().sample(n=min(2000, len(feat_df.dropna())), random_state=42),
        x="temperature_c", y="dew_point_estimate",
        color=feat_df["relative_humidity_pct"].dropna().sample(
            n=min(2000, len(feat_df.dropna())), random_state=42
        ),
        color_continuous_scale="Blues",
        title="Dew Point vs Temperature (colored by Humidity)",
        labels={"color": "Humidity (%)", "temperature_c": "Temperature (deg C)",
                "dew_point_estimate": "Dew Point Estimate (deg C)"},
        opacity=0.5,
    )
    apply_common_layout(fig_dew, height=400)
    st.plotly_chart(fig_dew, use_container_width=True)

# ── Section 5: Impact on Model Performance ───────────────────────────────────
st.header("5. How Each Feature Improves Prediction")

st.markdown(
    "Alright, let's actually measure the damage. We train a Random Forest to "
    "predict current temperature using different feature sets and watch how the "
    "MAE drops as we add engineered features one category at a time. This is "
    "the empirical proof that feature engineering is not just busywork."
)

# Build feature matrix
target = "temperature_c"
base_features = ["relative_humidity_pct", "wind_speed_kmh", "surface_pressure_hpa"]

feature_sets = {"Base (3 raw features)": base_features.copy()}

if datetime_feats:
    feature_sets["+ Datetime"] = base_features + datetime_feats
if lag_feats:
    feature_sets["+ Datetime + Lags"] = base_features + datetime_feats + lag_feats
if roll_feats:
    feature_sets["+ Datetime + Lags + Rolling"] = (
        base_features + datetime_feats + lag_feats + roll_feats
    )
if domain_feats:
    all_eng = base_features + datetime_feats + lag_feats + roll_feats + domain_feats
    feature_sets["+ All Engineered"] = all_eng

results = []
clean = feat_df.dropna()

if len(clean) > 100:
    for name, feats in feature_sets.items():
        available = [f for f in feats if f in clean.columns]
        if not available:
            continue
        X = clean[available].values
        y = clean[target].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        results.append({"Feature Set": name, "Num Features": len(available),
                         "MAE (deg C)": round(mae, 3), "R-squared": round(r2, 4)})

    res_df = pd.DataFrame(results)
    st.dataframe(res_df, use_container_width=True, hide_index=True)

    fig_imp = go.Figure()
    fig_imp.add_trace(go.Bar(
        x=res_df["Feature Set"], y=res_df["MAE (deg C)"],
        marker_color=["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E63946"][:len(res_df)],
    ))
    fig_imp.update_layout(yaxis_title="MAE (deg C)")
    apply_common_layout(fig_imp, "Model MAE by Feature Set", 400)
    st.plotly_chart(fig_imp, use_container_width=True)

    if len(results) >= 2:
        base_mae = results[0]["MAE (deg C)"]
        best_mae = results[-1]["MAE (deg C)"]
        improvement = ((base_mae - best_mae) / base_mae) * 100
        insight_box(
            f"Feature engineering reduced MAE from **{base_mae:.3f}** to "
            f"**{best_mae:.3f}** deg C -- a **{improvement:.1f}%** improvement. "
            "And we did not change the model at all! Same Random Forest, same "
            "hyperparameters. The only difference is better inputs. Lag features "
            "typically provide the biggest single boost, because temperature is "
            "strongly autocorrelated."
        )
else:
    st.warning("Not enough data after filtering. Broaden your date range in the sidebar.")

warning_box(
    "Lag features create a subtle but serious **data leakage** risk. The lag_24h "
    "feature is safe for forecasting 'what will the temperature be 24 hours from now?' "
    "because you *do* know the temperature 24 hours ago at prediction time. But if "
    "you are forecasting 1 hour ahead and use lag_1h, you are using information from "
    "the future. Always ask: 'Would I have this feature available at the moment I "
    "need to make the prediction?'"
)

code_example("""
import numpy as np

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Lag features
df['temp_lag_24h'] = df['temperature_c'].shift(24)

# Rolling statistics
df['temp_rolling_24h_mean'] = df['temperature_c'].rolling(24).mean()

# Domain feature: dew point estimate
df['dew_point'] = df['temperature_c'] - (100 - df['relative_humidity_pct']) / 5
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "Why is cyclical encoding (sin/cos) better than raw integers for hour-of-day?",
    [
        "It reduces the number of features from 24 to 2",
        "It preserves the circular nature: hour 23 is close to hour 0",
        "It makes the data normally distributed",
        "It removes outliers",
    ],
    1,
    "Raw integer encoding treats 23 and 0 as far apart (23 units of difference). "
    "Sin/cos encoding maps them to nearby points on a circle, preserving the "
    "physical reality that midnight and 11 PM are one hour apart.",
    key="fe_quiz",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "Feature engineering transforms raw data into informative model inputs. It is often the highest-leverage thing you can do to improve a model.",
    "Cyclical encoding (sin/cos) is essential for periodic features like hour and month. Without it, algorithms think midnight and 11 PM are far apart.",
    "Lag features capture temporal dependencies (e.g., lag-24h captures the diurnal pattern) and are often the single most valuable engineered feature for weather.",
    "Rolling statistics smooth out noise and capture local trends -- they tell the model 'what has the weather been doing lately?'",
    "Domain features (dew point, heat index) encode physical knowledge that would take the model many splits or parameters to learn on its own.",
    "Even simple engineered features can dramatically reduce prediction error -- same model, better inputs, better results.",
])
