"""Chapter 40 -- Scaling & Encoding."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ml_helpers import prepare_classification_data, classification_metrics
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(40, "Scaling & Encoding", part="IX")

st.markdown("""
Let me tell you about a bug that has wasted more collective hours of data scientist time than
perhaps any other single mistake in the field.

You have a dataset with four columns: temperature in Celsius, humidity as a percentage, wind speed
in km/h, and surface pressure in hectopascals. You want to build a model that, given a single
hourly weather reading -- just those four numbers -- can guess which city it came from. This is
a genuinely interesting problem! Dallas in July feels nothing like NYC in January. Houston is
a swamp. LA is... LA. The weather signatures of these cities are real and distinctive, and a
good model should be able to tell them apart.

So you reach for K-Nearest Neighbors, which is the simplest possible approach: to classify a new
reading, just find the K most similar readings in the training set and go with the majority vote.
"Similar" means "close in 4-dimensional space" -- small Euclidean distance across all four features.

You train it. You get 35% accuracy. On a 6-class problem where random guessing gives you ~17%,
this is... technically better than chance? But it's terrible. What went wrong?

Here is what went wrong. Your four features are:
- **Temperature**: ranges from roughly -5 to 45. Call it a spread of ~50.
- **Humidity**: ranges from roughly 5 to 100. Call it a spread of ~95.
- **Wind speed**: ranges from 0 to maybe 60. Call it a spread of ~60.
- **Surface pressure**: ranges from roughly 950 to 1050. Call it a spread of ~100 -- but *centered around 1000*.

Now think about what Euclidean distance actually computes. It's the square root of the sum of
squared differences across all features. If Reading A has pressure=1010 and Reading B has
pressure=1020, that's a difference of 10, contributing 100 to the sum of squares. If Reading A
has temperature=15 and Reading B has temperature=25, that's also a difference of 10, also
contributing 100 to the sum of squares.

But those two "10s" are completely different things! A 10-degree temperature difference is *enormous*
-- it's the difference between wearing a winter coat and wearing a t-shirt. A 10 hPa pressure
difference is barely noticeable -- you need a barometer to even detect it. Yet the distance formula
treats them identically.

Worse: because pressure values hover around 1000 while temperature values hover around 20, the
*absolute variation* in pressure numerically dwarfs everything else. The model is essentially
computing "which training point has the most similar pressure?" and ignoring the other three
features entirely. It's like trying to evaluate restaurants based on food quality, service,
ambiance, and zip code -- the zip code, being a 5-digit number, would dominate the rating
unless you did something about the scales.

Scaling does something about the scales.
""")

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

concept_box(
    "The Core Insight",
    "Machine learning algorithms that compute distances between data points -- KNN, SVM, "
    "K-Means, neural networks with gradient descent -- are sensitive to the numeric scale "
    "of each feature. If one feature is measured in units that happen to produce large numbers "
    "(like pressure in hPa, ~1000) and another is measured in units that produce small numbers "
    "(like temperature in Celsius, ~20), the large-number feature will dominate distance "
    "calculations. The fix is simple: transform all features to comparable scales before "
    "feeding them to the model. This is called <b>feature scaling</b>, and it is one of "
    "those things that sounds too simple to matter until you see the accuracy jump by 20+ "
    "percentage points."
)

# ── Section 1: The Problem -- Unscaled Features ─────────────────────────────
st.header("1. Let's Actually Look at the Numbers")

st.markdown("""
Before I show you the fix, let me make the problem visceral. Here are the actual summary
statistics for our four weather features. Pay attention to the scales.
""")

sub = fdf.dropna(subset=FEATURE_COLS).sample(n=min(5000, len(fdf)), random_state=42)

# Show raw feature ranges
ranges = sub[FEATURE_COLS].describe().loc[["min", "max", "mean", "std"]].T
ranges.index = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS]
st.dataframe(ranges.round(2), use_container_width=True)

st.markdown("""
See it? Temperature lives in the range of roughly -5 to 45. Wind speed from 0 to maybe 60.
But surface pressure? Its *mean* is around 1010. Even its standard deviation (~8 hPa) is tiny
relative to its mean. When you compute Euclidean distance between two readings, the squared
difference in pressure will be something like (1012 - 1005)^2 = 49, while the squared difference
in temperature might be (22 - 18)^2 = 16. Pressure wins, even though that 4-degree temperature
difference tells you much more about which city you're in.

Here's what the fix looks like -- StandardScaler transforms each feature to have mean=0 and
standard deviation=1:
""")

fig_box = make_subplots(rows=1, cols=2, subplot_titles=["Raw Features (look at those scales!)", "After StandardScaler (all on same footing)"])

for feat in FEATURE_COLS:
    fig_box.add_trace(go.Box(
        y=sub[feat].values[:2000], name=FEATURE_LABELS.get(feat, feat)[:12],
        showlegend=False,
    ), row=1, col=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(sub[FEATURE_COLS])
for i, feat in enumerate(FEATURE_COLS):
    fig_box.add_trace(go.Box(
        y=X_scaled[:2000, i], name=FEATURE_LABELS.get(feat, feat)[:12],
        showlegend=False,
    ), row=1, col=2)

fig_box.update_layout(height=450, template="plotly_white")
st.plotly_chart(fig_box, use_container_width=True)

insight_box(
    "In the left panel, pressure is so far from the other features that they're essentially "
    "invisible at that y-axis scale. After StandardScaler (right panel), every feature has "
    "mean=0 and std=1. Now a 1-unit difference in scaled temperature carries the same weight "
    "as a 1-unit difference in scaled pressure. The model can actually *see* all four features."
)

# ── Section 2: Comparing Scalers ─────────────────────────────────────────────
st.header("2. Three Ways to Scale (and When to Use Each)")

st.markdown("""
There are several standard approaches to scaling, and which one you pick actually matters
sometimes. Let me walk through the three you'll encounter most often.
""")

st.sidebar.subheader("Scaling Settings")
scaler_choice = st.sidebar.selectbox(
    "Scaler to Visualize",
    ["StandardScaler", "MinMaxScaler", "RobustScaler"],
    key="scaler_choice",
)

concept_box(
    "The Three Scalers, Explained Like You're a Curious Human",
    "<b>StandardScaler (z-score normalization)</b>: Take each value, subtract the mean of that "
    "feature, and divide by its standard deviation. The result: every feature has mean=0 and "
    "std=1. This is the default choice. It works well when your features are roughly normally "
    "distributed, which weather data largely is (temperature within a city-month is approximately "
    "Gaussian -- we showed this back in Chapter 3).<br><br>"
    "<b>MinMaxScaler</b>: Take each value and squish it into the range [0, 1] by subtracting "
    "the minimum and dividing by the range. Simple and intuitive. The catch: if you have one "
    "freak reading where Dallas hit -15 C during a polar vortex, that single outlier stretches "
    "the range, compressing all the normal readings into a narrower band. One bad data point "
    "can distort the scaling for everyone else.<br><br>"
    "<b>RobustScaler</b>: Same idea as StandardScaler but uses the median instead of the mean "
    "and the interquartile range (IQR) instead of the standard deviation. Because the median "
    "and IQR are not affected by extreme values, this scaler shrugs off outliers. If your data "
    "has weird readings (and weather data always does -- storms, sensor glitches, polar vortices), "
    "RobustScaler keeps the scaling sensible."
)

formula_box(
    "StandardScaler",
    r"\underbrace{z}_{\text{scaled value}} = \frac{\underbrace{x}_{\text{raw feature}} - \underbrace{\mu}_{\text{feature mean}}}{\underbrace{\sigma}_{\text{feature std dev}}}",
    "Subtracts the mean, divides by std. The workhorse. Use this unless you have a reason not to."
)
formula_box(
    "MinMaxScaler",
    r"\underbrace{x_{scaled}}_{\text{scaled to [0,1]}} = \frac{\underbrace{x}_{\text{raw feature}} - \underbrace{x_{min}}_{\text{feature minimum}}}{\underbrace{x_{max} - x_{min}}_{\text{feature range}}}",
    "Maps everything to [0, 1]. Clean and simple, but one outlier can compress the whole range."
)
formula_box(
    "RobustScaler",
    r"\underbrace{x_{scaled}}_{\text{robust scaled}} = \frac{\underbrace{x}_{\text{raw feature}} - \underbrace{\text{median}}_{\text{middle value}}}{\underbrace{\text{IQR}}_{\text{interquartile range}}}",
    "Uses the median and interquartile range. Outliers don't warp the scaling for normal observations."
)

scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
}

# Show chosen scaler's effect on each feature
chosen_scaler = scalers[scaler_choice]
X_raw = sub[FEATURE_COLS].values
X_transformed = chosen_scaler.fit_transform(X_raw)

fig_hist = make_subplots(
    rows=2, cols=2,
    subplot_titles=[FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
)
positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
colors = ["#2E86C1", "#E63946", "#2A9D8F", "#F4A261"]
for i, (feat, pos) in enumerate(zip(FEATURE_COLS, positions)):
    fig_hist.add_trace(go.Histogram(
        x=X_transformed[:, i], nbinsx=50,
        marker_color=colors[i], showlegend=False,
        opacity=0.7,
    ), row=pos[0], col=pos[1])

fig_hist.update_layout(height=500, template="plotly_white",
                        title_text=f"Feature Distributions After {scaler_choice}")
st.plotly_chart(fig_hist, use_container_width=True)

# Side-by-side comparison table
comp_data = []
for name, sc in scalers.items():
    Xt = sc.fit_transform(X_raw)
    comp_data.append({
        "Scaler": name,
        "Min (across features)": f"{Xt.min():.2f}",
        "Max (across features)": f"{Xt.max():.2f}",
        "Mean (avg)": f"{Xt.mean():.4f}",
        "Std (avg)": f"{Xt.std():.4f}",
    })
comp_df = pd.DataFrame(comp_data)
st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ── Section 3: KNN With and Without Scaling ──────────────────────────────────
st.header("3. The Experiment: Predicting City from a Single Weather Reading")

st.markdown("""
OK, enough theory. Let's run the actual experiment.

Here's the setup: we take each hourly weather reading -- a single row of data with four numbers
(temperature, humidity, wind speed, pressure) -- and we ask: **can a KNN classifier figure out
which of the 6 cities this reading came from?**

Why is this even possible? Because cities have weather fingerprints. Houston is warm and humid.
LA is mild and dry. NYC gets cold in winter. Dallas has wild temperature swings. These patterns
are real, and they show up in the data. A reading of 35°C with 80% humidity screams "Houston in
August." A reading of 22°C with 25% humidity whispers "Los Angeles, any month."

KNN works by finding the K most similar readings in the training set (similar = smallest Euclidean
distance across all four features) and taking a majority vote of their city labels. It's
delightfully simple. And it's the *perfect* algorithm to demonstrate why scaling matters, because
it uses raw distances, and raw distances are exactly where unscaled features cause havoc.

Adjust K below and watch what happens:
""")

k = st.slider("Number of neighbors (K)", 3, 21, 5, 2, key="knn_k")

# Prepare data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
clean = sub.dropna(subset=FEATURE_COLS)
X_raw_knn = clean[FEATURE_COLS].values
y_knn = le.fit_transform(clean["city"])

X_train_r, X_test_r, y_train, y_test = train_test_split(
    X_raw_knn, y_knn, test_size=0.2, random_state=42, stratify=y_knn,
)

# Unscaled KNN
knn_unscaled = KNeighborsClassifier(n_neighbors=k)
knn_unscaled.fit(X_train_r, y_train)
acc_unscaled = accuracy_score(y_test, knn_unscaled.predict(X_test_r))

# Scaled KNN -- all three scalers
knn_results = [{"Scaler": "None (raw)", "Accuracy": round(acc_unscaled, 4)}]
for name, sc in scalers.items():
    Xtr = sc.fit_transform(X_train_r)
    Xte = sc.transform(X_test_r)
    knn_s = KNeighborsClassifier(n_neighbors=k)
    knn_s.fit(Xtr, y_train)
    acc_s = accuracy_score(y_test, knn_s.predict(Xte))
    knn_results.append({"Scaler": name, "Accuracy": round(acc_s, 4)})

knn_df = pd.DataFrame(knn_results)

fig_knn = go.Figure()
colors_bar = ["#AAB7B8", "#2E86C1", "#E63946", "#2A9D8F"]
fig_knn.add_trace(go.Bar(
    x=knn_df["Scaler"], y=knn_df["Accuracy"],
    marker_color=colors_bar,
    text=knn_df["Accuracy"].apply(lambda x: f"{x:.1%}"),
    textposition="outside",
))
fig_knn.update_layout(yaxis_title="Accuracy", yaxis_range=[0, 1.05])
apply_common_layout(fig_knn, f"KNN (K={k}): Can It Guess the City from One Weather Reading?", 400)
st.plotly_chart(fig_knn, use_container_width=True)

improvement = ((knn_results[1]["Accuracy"] - acc_unscaled) / acc_unscaled) * 100 if acc_unscaled > 0 else 0

st.markdown(f"""
Let that sink in.

Without scaling, KNN gets **{acc_unscaled:.1%}** accuracy. It's doing barely better than random
guessing (which would give ~17% on 6 classes). The model is essentially computing "which training
reading has the most similar *pressure*?" and ignoring temperature, humidity, and wind speed,
because those features' values are numerically tiny compared to pressure.

With StandardScaler, accuracy jumps to **{knn_results[1]['Accuracy']:.1%}**. Same algorithm. Same
data. Same K. The *only* difference is one line of preprocessing code. This is a ~{abs(improvement):.0f}%
improvement, and it comes from doing nothing more than putting all features on the same scale.

This is not a weird edge case. This is what happens *by default* if you forget to scale, and it
happens with any distance-based algorithm. KNN, SVM, K-Means, neural networks -- they all suffer
from this. Tree-based models (Random Forest, XGBoost) don't, for reasons we'll get to in a moment.
""")

warning_box(
    "A subtle but important point: you must fit the scaler on the **training data only**, then "
    "use that fitted scaler to transform both training and test data. If you fit the scaler on "
    "the full dataset (including test data), the scaler 'knows' the test set's mean and standard "
    "deviation. This is a form of data leakage -- the model has indirect access to test-set "
    "information that it shouldn't have. In practice, use an sklearn Pipeline (shown below) to "
    "make this automatic and impossible to screw up."
)

# ── Section 4: One-Hot Encoding ──────────────────────────────────────────────
st.header("4. The Other Problem: Categorical Variables")

st.markdown("""
Scaling handles the numeric features, but what about categorical ones? Our dataset has a `city`
column with values like "Houston" and "Los Angeles," and a `season` column with values like
"Summer" and "Winter."

Most ML algorithms can't handle strings. They want numbers. So you might be tempted to encode
cities as integers: Dallas=0, Houston=1, LA=2, etc. But this creates a subtle problem: the model
will treat these as *ordered* numbers, implying that Houston (1) is "between" Dallas (0) and
LA (2), and that the "distance" from Dallas to LA is twice the distance from Dallas to Houston.
This is obviously nonsense -- there's no meaningful numeric ordering of cities.
""")

concept_box(
    "One-Hot Encoding: The Standard Fix",
    "Instead of mapping categories to integers, create a separate binary column for each "
    "category. Each column is 1 if the observation belongs to that category, 0 otherwise. "
    "So 'Houston' becomes [0, 1, 0, 0, 0, 0] (if the columns are ordered alphabetically "
    "as Austin, Houston, LA, NYC, Dallas, San Antonio). No implied ordering, no fake distances. "
    "The model can learn separate effects for each city without assuming any relationship between them."
)

st.markdown("**Here's what one-hot encoding looks like on real data -- the `city` column:**")

ohe_example = fdf[["city"]].head(12).copy()
ohe_result = pd.get_dummies(ohe_example, columns=["city"], prefix="", prefix_sep="")
st.dataframe(ohe_result, use_container_width=True)

st.markdown("**And the `season` column:**")
season_ohe = pd.get_dummies(fdf[["season"]].head(12), columns=["season"], prefix="", prefix_sep="")
st.dataframe(season_ohe, use_container_width=True)

warning_box(
    "With K categories, one-hot encoding creates K binary columns. For linear models (logistic "
    "regression, linear regression), this creates a problem called perfect multicollinearity: "
    "the K columns always sum to exactly 1, which is perfectly correlated with the intercept "
    "term. The fix is to drop one column with `drop_first=True` -- the dropped category becomes "
    "the 'reference' that all others are compared against. Tree-based models don't have this "
    "problem because they split on individual features one at a time."
)

# ── Section 5: Scaling + Encoding Pipeline ───────────────────────────────────
st.header("5. The Right Way to Do This in Practice")

st.markdown("""
In a real project, you don't want to manually scale numeric features, one-hot encode categorical
features, remember to fit on training data only, and hope you don't make a mistake. You use an
sklearn Pipeline, which packages all of this into a single object:
""")

code_example("""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

numeric_features = ['temperature_c', 'relative_humidity_pct',
                     'wind_speed_kmh', 'surface_pressure_hpa']
categorical_features = ['season']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features),
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5)),
])

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
""")

st.markdown("""
The Pipeline handles everything: it fits the scaler and encoder on training data only, applies
them consistently during prediction, and packages the whole thing into one serializable object.
If you're not using pipelines in your ML code, you're accumulating technical debt and creating
opportunities for subtle data leakage bugs. Use them.
""")

# ── Section 6: When Not to Scale ─────────────────────────────────────────────
st.header("6. Models That Don't Need Scaling (and Why)")

st.markdown("""
Here's a question you might be asking: if scaling is so important, why didn't we mention it
in every previous chapter?

The answer is that **tree-based models are scale-invariant**. A decision tree asks questions
like "is temperature > 25°C?" and "is pressure > 1015 hPa?" -- it splits on one feature at a
time, using the actual values. It never computes distances between multi-feature data points.
So whether pressure is in hPa (around 1000) or atmospheres (around 1.0) makes zero difference
to the tree's splits.

This is, incidentally, one of the reasons Random Forest and XGBoost are so popular in practice:
they just work out of the box, without the preprocessing fussiness that distance-based models
demand.
""")

st.markdown("""
| **Model** | **Needs Scaling?** | **Why** |
|---|---|---|
| KNN | Yes | Euclidean distance treats all features equally by magnitude |
| SVM | Yes | Kernel functions compute distances; unscaled features distort the kernel |
| Linear/Logistic Regression | Recommended | Coefficients are only comparable across features if features are on the same scale |
| Decision Tree / Random Forest | No | Splits on individual features using thresholds; scale doesn't affect split quality |
| XGBoost / Gradient Boosting | No | Tree-based, same reason as above |
| Neural Networks | Yes | Gradient descent converges much faster when inputs are on similar scales |
""")

insight_box(
    "This table explains a lot about the ML ecosystem. Tree-based models (Random Forest, "
    "XGBoost, LightGBM) dominate Kaggle competitions and industry applications partly because "
    "they don't need scaling, don't need one-hot encoding (they can split on ordinal integers "
    "just fine), and are generally robust to the kind of preprocessing mistakes that tank "
    "distance-based models. The tradeoff: they can't do things like capture smooth decision "
    "boundaries the way SVMs or neural networks can."
)

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "You train a KNN classifier on two features: 'temperature_c' (range: 0-40) and "
    "'surface_pressure_hpa' (range: 950-1050). Without scaling, the model gets 30% "
    "accuracy. You add StandardScaler and accuracy jumps to 55%. What was happening?",
    [
        "KNN was overfitting to the training data",
        "Pressure dominated the Euclidean distance, making the model essentially ignore temperature",
        "The model needed more neighbors (larger K)",
        "StandardScaler added new information that wasn't in the raw data",
    ],
    1,
    "Pressure values are ~1000 while temperature values are ~20. In raw Euclidean distance, "
    "a trivial 5 hPa pressure difference contributes more to the distance than a massive "
    "20-degree temperature difference. The model was doing 'nearest pressure neighbor' not "
    "'nearest weather neighbor.' Scaling puts both features on equal footing so the model "
    "can actually use temperature -- which is the most informative feature for distinguishing "
    "cities.",
    key="scaling_quiz",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "Feature scaling is not optional for distance-based algorithms (KNN, SVM, K-Means, neural nets). Without it, features measured in large units dominate the distance metric and other features become invisible to the model.",
    "StandardScaler (z-score normalization) is the default choice. It transforms each feature to mean=0, std=1. Use MinMaxScaler when you need [0,1] bounds, and RobustScaler when outliers would distort the mean and standard deviation.",
    "Always fit the scaler on training data only, then transform both train and test with the fitted scaler. Fitting on the full dataset is data leakage. Use sklearn Pipeline to automate this.",
    "Categorical variables (city names, seasons) must be one-hot encoded for most algorithms. Integer encoding implies a false ordering. Use `drop_first=True` for linear models to avoid multicollinearity.",
    "Tree-based models (Random Forest, XGBoost, LightGBM) do NOT need scaling -- they split on individual features using thresholds, so the absolute scale of values doesn't matter. This is a major practical advantage.",
    "Use sklearn Pipeline to combine scaling, encoding, and modeling into one reproducible, leak-proof object. This is the standard in production ML and it prevents an entire category of bugs.",
])
