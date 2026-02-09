"""Chapter 26: Naive Bayes -- Priors, likelihoods, and the independence assumption."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.naive_bayes import GaussianNB
from scipy.stats import norm

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.ml_helpers import prepare_classification_data, classification_metrics, plot_confusion_matrix
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS, FEATURE_UNITS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(26, "Naive Bayes", part="V")
st.markdown(
    "Naive Bayes is one of those beautiful lies that turns out to be useful anyway. "
    "It applies **Bayes' theorem** -- the crown jewel of probability theory -- with "
    "the brazenly false assumption that all your features are independent. Temperature "
    "and humidity? Independent, says Naive Bayes with a straight face. This is obviously "
    "wrong. And yet the classifier works surprisingly well, has zero hyperparameters to "
    "tune, trains in milliseconds, and gives you interpretable probability estimates. "
    "Sometimes the wrong model is the right tool."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
filt = sidebar_filters(df)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 -- Bayes' Theorem
# ══════════════════════════════════════════════════════════════════════════════
st.header("1. Bayes' Theorem")

concept_box(
    "From Priors to Posteriors",
    "Bayes' theorem is really just a formalization of how rational belief updating "
    "works. You start with a <b>prior</b> -- your belief before seeing any evidence. "
    "(Before checking the weather, Dallas and Houston are each about 1/6 of our "
    "dataset, so each has a prior of ~0.17.) Then you observe some features and compute "
    "the <b>likelihood</b> -- how probable those features are for each city. Multiply "
    "them together, normalize, and you get the <b>posterior</b> -- your updated belief. "
    "It is Bayesian reasoning, and it is the mathematically correct way to update beliefs "
    "with evidence."
)

formula_box(
    "Bayes' Theorem for Classification",
    r"P(\text{city} \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid \text{city}) \cdot P(\text{city})}{P(\mathbf{x})}",
    "Posterior = (Likelihood x Prior) / Evidence. The evidence term P(x) is the same for "
    "all cities, so in practice we just compare the numerators and pick the winner."
)

formula_box(
    "Naive Assumption (Feature Independence)",
    r"P(\mathbf{x} \mid \text{city}) = \prod_{j=1}^{p} P(x_j \mid \text{city})",
    "This is the 'naive' part: we assume each feature is independent given the class. "
    "So instead of estimating the joint distribution of all features (which requires "
    "exponentially many parameters), we just estimate each feature's distribution "
    "separately. It is wrong, but it makes the math tractable and the estimates stable."
)

# Show priors
st.subheader("Prior Probabilities (from data)")
prior_counts = filt["city"].value_counts()
prior_probs = prior_counts / prior_counts.sum()
prior_df = pd.DataFrame({
    "City": prior_probs.index,
    "Count": prior_counts.values,
    "P(city)": prior_probs.values,
}).reset_index(drop=True)
prior_df["P(city)"] = prior_df["P(city)"].round(4)

col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(prior_df, use_container_width=True)
with col2:
    fig_prior = px.bar(
        prior_df, x="City", y="P(city)",
        title="Prior Probabilities",
        color="City", color_discrete_map=CITY_COLORS,
        text_auto=".3f"
    )
    apply_common_layout(fig_prior, title="Prior Probabilities", height=350)
    fig_prior.update_layout(showlegend=False)
    st.plotly_chart(fig_prior, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 -- Gaussian Likelihoods
# ══════════════════════════════════════════════════════════════════════════════
st.header("2. Gaussian Likelihoods")

concept_box(
    "Gaussian Naive Bayes",
    "For continuous features like temperature, Gaussian NB assumes each feature follows "
    "a normal (bell-curve) distribution within each class. So 'temperature given NYC' "
    "is a bell curve centered around NYC's average temperature, and 'temperature given "
    "LA' is a different bell curve centered around LA's average temperature.<br><br>"
    "The model just needs to estimate two numbers per feature per city: the <b>mean</b> "
    "and the <b>standard deviation</b>. That is 4 features x 6 cities x 2 parameters = "
    "48 numbers total. The entire model fits in a tweet."
)

# Show the Gaussian distributions per feature per city
feature_to_show = st.selectbox("Select feature to visualize likelihoods", FEATURE_COLS, key="nb_feat")

fig_lik = go.Figure()
x_range = np.linspace(filt[feature_to_show].min() - 5, filt[feature_to_show].max() + 5, 300)

city_stats = {}
for city in CITY_LIST:
    city_data = filt[filt["city"] == city][feature_to_show].dropna()
    mu = city_data.mean()
    sigma = city_data.std()
    city_stats[city] = (mu, sigma)
    y_pdf = norm.pdf(x_range, mu, sigma)
    fig_lik.add_trace(go.Scatter(
        x=x_range, y=y_pdf, mode="lines", name=city,
        line=dict(color=CITY_COLORS[city], width=2)
    ))

apply_common_layout(fig_lik, title=f"Gaussian Likelihoods: P({FEATURE_LABELS[feature_to_show]} | city)", height=450)
fig_lik.update_layout(
    xaxis_title=FEATURE_LABELS[feature_to_show],
    yaxis_title="Probability Density",
)
st.plotly_chart(fig_lik, use_container_width=True)

# Show stats table
stats_table = pd.DataFrame({
    city: {"Mean": f"{mu:.2f}", "Std Dev": f"{sigma:.2f}"}
    for city, (mu, sigma) in city_stats.items()
}).T
st.markdown(f"**Per-city statistics for {FEATURE_LABELS[feature_to_show]}:**")
st.dataframe(stats_table, use_container_width=True)

insight_box(
    "Where the Gaussian curves overlap, the classifier is uncertain -- an observation in "
    "that region could plausibly come from either city. Where the curves are well-separated "
    "(like LA vs NYC on temperature), classification is easy. The Texas cities, predictably, "
    "have overlapping curves for almost everything."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 -- Step-by-step P(city|features) Calculation
# ══════════════════════════════════════════════════════════════════════════════
st.header("3. Step-by-Step Posterior Calculation")

st.markdown(
    "This is the most transparent classification algorithm you will ever see. Input "
    "some weather values below and watch the math happen: we compute the prior, "
    "multiply by the likelihood of each feature, and out comes a posterior probability "
    "for each city. No hidden layers, no black boxes, just Bayes' theorem in action."
)

# Train the model to get the learned parameters
gnb = GaussianNB()
X_all = filt[FEATURE_COLS].dropna()
y_all = filt.loc[X_all.index, "city"]
gnb.fit(X_all, y_all)

# Input sliders
st.subheader("Input Weather Observation")
col_inputs = st.columns(2)
with col_inputs[0]:
    input_temp = st.slider(
        f"Temperature ({FEATURE_UNITS['temperature_c']})",
        float(filt["temperature_c"].min()), float(filt["temperature_c"].max()),
        float(filt["temperature_c"].mean()), 0.5, key="nb_temp"
    )
    input_humid = st.slider(
        f"Relative Humidity ({FEATURE_UNITS['relative_humidity_pct']})",
        float(filt["relative_humidity_pct"].min()), float(filt["relative_humidity_pct"].max()),
        float(filt["relative_humidity_pct"].mean()), 1.0, key="nb_humid"
    )
with col_inputs[1]:
    input_wind = st.slider(
        f"Wind Speed ({FEATURE_UNITS['wind_speed_kmh']})",
        float(filt["wind_speed_kmh"].min()), float(filt["wind_speed_kmh"].max()),
        float(filt["wind_speed_kmh"].mean()), 0.5, key="nb_wind"
    )
    input_press = st.slider(
        f"Surface Pressure ({FEATURE_UNITS['surface_pressure_hpa']})",
        float(filt["surface_pressure_hpa"].min()), float(filt["surface_pressure_hpa"].max()),
        float(filt["surface_pressure_hpa"].mean()), 0.5, key="nb_press"
    )

input_values = np.array([[input_temp, input_humid, input_wind, input_press]])

# Get probabilities from sklearn
proba = gnb.predict_proba(input_values)[0]
predicted_city = gnb.predict(input_values)[0]

# Manual step-by-step calculation
st.subheader("Calculation Breakdown")

classes = gnb.classes_
log_posteriors = {}
for i, city in enumerate(classes):
    prior = gnb.class_prior_[i]
    log_prior = np.log(prior)

    log_likelihoods = []
    likelihood_strs = []
    for j, feat in enumerate(FEATURE_COLS):
        mu = gnb.theta_[i, j]
        sigma = np.sqrt(gnb.var_[i, j])
        val = input_values[0, j]
        ll = norm.logpdf(val, mu, sigma)
        log_likelihoods.append(ll)
        p_val = norm.pdf(val, mu, sigma)
        likelihood_strs.append(f"P({FEATURE_LABELS[feat]}={val:.1f} | {city}) = {p_val:.6f}")

    total_log_posterior = log_prior + sum(log_likelihoods)
    log_posteriors[city] = total_log_posterior

# Normalize using log-sum-exp
max_log = max(log_posteriors.values())
posteriors = {city: np.exp(lp - max_log) for city, lp in log_posteriors.items()}
total = sum(posteriors.values())
posteriors = {city: p / total for city, p in posteriors.items()}

# Display posteriors as bar chart
post_df = pd.DataFrame({
    "City": list(posteriors.keys()),
    "P(city|features)": list(posteriors.values()),
}).sort_values("P(city|features)", ascending=True)

fig_post = px.bar(
    post_df, x="P(city|features)", y="City", orientation="h",
    title=f"Posterior Probabilities -- Predicted: {predicted_city}",
    color="City", color_discrete_map=CITY_COLORS,
    text_auto=".3f"
)
apply_common_layout(fig_post, title=f"Posterior Probabilities -- Predicted: {predicted_city}", height=400)
fig_post.update_layout(showlegend=False)
st.plotly_chart(fig_post, use_container_width=True)

st.success(f"**Predicted city: {predicted_city}** with probability {posteriors[predicted_city]:.3f}")

# Show detailed calculation for top city
with st.expander("Show detailed calculation for each city"):
    for i, city in enumerate(classes):
        prior = gnb.class_prior_[i]
        st.markdown(f"**{city}** (prior = {prior:.4f})")
        for j, feat in enumerate(FEATURE_COLS):
            mu = gnb.theta_[i, j]
            sigma = np.sqrt(gnb.var_[i, j])
            val = input_values[0, j]
            p_val = norm.pdf(val, mu, sigma)
            st.markdown(
                f"  - P({FEATURE_LABELS[feat]}={val:.1f} | {city}) = "
                f"N({val:.1f}; mu={mu:.2f}, sigma={sigma:.2f}) = **{p_val:.6f}**"
            )
        st.markdown(f"  - **Posterior: {posteriors[city]:.4f}**")
        st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 -- Model Evaluation
# ══════════════════════════════════════════════════════════════════════════════
st.header("4. Model Evaluation: Full 6-City Classification")

X_train, X_test, y_train, y_test, le, scaler = prepare_classification_data(
    filt, FEATURE_COLS, target="city", test_size=0.2, scale=False, seed=42
)
labels = le.classes_.tolist()

gnb_eval = GaussianNB()
gnb_eval.fit(X_train, y_train)
y_pred = gnb_eval.predict(X_test)
metrics = classification_metrics(y_test, y_pred, labels=labels)

col1, col2 = st.columns(2)
col1.metric("Test Accuracy", f"{metrics['accuracy']:.1%}")
col2.metric("Test Samples", len(y_test))

st.plotly_chart(
    plot_confusion_matrix(metrics["confusion_matrix"], labels),
    use_container_width=True
)

st.markdown("**Per-class metrics:**")
report_df = pd.DataFrame(metrics["report"]).T
report_df = report_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
report_df = report_df[["precision", "recall", "f1-score", "support"]].round(3)
st.dataframe(report_df, use_container_width=True)

warning_box(
    "The independence assumption means Naive Bayes pretends temperature and humidity are "
    "unrelated. They are not -- hot air holds more moisture. This violated assumption "
    "does not destroy the model's ability to rank classes correctly (the right city "
    "usually still wins), but it does make the probability estimates overconfident. "
    "When Naive Bayes says '99% NYC,' it might really be 85% NYC. The ranking is "
    "trustworthy; the calibration is not."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 -- The Independence Assumption
# ══════════════════════════════════════════════════════════════════════════════
st.header("5. How Naive is the Independence Assumption?")

st.markdown(
    "Let us be honest about the lie at the heart of Naive Bayes. The model assumes "
    "all features are independent given the class. Let us check how badly violated "
    "this assumption is by looking at the correlation matrix."
)

corr = filt[FEATURE_COLS].corr()
corr_display = corr.rename(columns=FEATURE_LABELS, index=FEATURE_LABELS)

fig_corr = px.imshow(
    corr_display.values,
    x=corr_display.columns.tolist(),
    y=corr_display.index.tolist(),
    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    title="Feature Correlation Matrix",
    text_auto=".2f", aspect="auto"
)
apply_common_layout(fig_corr, title="Feature Correlation Matrix", height=400)
st.plotly_chart(fig_corr, use_container_width=True)

max_corr_val = corr.abs().where(np.triu(np.ones_like(corr, dtype=bool), k=1)).max().max()
insight_box(
    f"The maximum absolute correlation between features is **{max_corr_val:.2f}**. "
    "So the features are correlated, and the independence assumption is wrong. And yet "
    "Naive Bayes still works. This is one of those results that bothered statisticians "
    "for decades: the model is theoretically wrong but practically useful. The reason "
    "it works is that classification only requires getting the *ranking* of posteriors "
    "right, not their exact values. Even with correlated features, the right city "
    "usually has the highest posterior."
)

code_example("""
from sklearn.naive_bayes import GaussianNB

# Gaussian NB -- no hyperparameters to tune!
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict probabilities (not just class labels)
probas = gnb.predict_proba(X_test)

# Access learned parameters
print("Class priors:", gnb.class_prior_)
print("Class means:", gnb.theta_)        # shape: (n_classes, n_features)
print("Class variances:", gnb.var_)       # shape: (n_classes, n_features)

# Step-by-step: for a new observation [temp, humid, wind, press]
# P(city | x) ∝ P(city) * ∏ N(x_j; mu_j, sigma_j)
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 -- Quiz & Takeaways
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

quiz(
    "Why is Naive Bayes called 'naive'?",
    [
        "Because it uses a simple algorithm",
        "Because it assumes all features are independent given the class",
        "Because it only works with small datasets",
        "Because it ignores the prior probability",
    ],
    correct_idx=1,
    explanation="The 'naive' part is the assumption that features are conditionally independent "
    "given the class label. Temperature and humidity are clearly correlated, so the assumption "
    "is clearly wrong. But the model shrugs and works anyway.",
    key="q_nb_1"
)

quiz(
    "In Gaussian Naive Bayes, what parameters are estimated for each feature per class?",
    [
        "Only the mean",
        "Mean and variance (mu and sigma^2)",
        "Median and IQR",
        "Min and max",
    ],
    correct_idx=1,
    explanation="A Gaussian distribution is fully specified by its mean and variance. That is "
    "all Naive Bayes needs: 2 numbers per feature per class. The entire model is just a "
    "table of means and variances.",
    key="q_nb_2"
)

takeaways([
    "Naive Bayes applies Bayes' theorem with a boldly wrong assumption: feature independence. It is the useful lie at the heart of one of ML's most practical algorithms.",
    "Gaussian NB models each feature as a normal distribution per class -- just mean and variance. The whole model is 48 numbers for our dataset.",
    "Despite the violated independence assumption, the classification ranking is usually correct. The probabilities are overconfident, but the winner is right.",
    "No hyperparameters to tune. Trains in milliseconds. This is the algorithm you reach for when you want a fast baseline or interpretable probabilities.",
    "The posterior P(city|features) is proportional to the prior times the product of likelihoods. You can compute it by hand on the back of a napkin.",
    "When features are highly correlated, the probability estimates become poorly calibrated -- the model is too confident. Use with appropriate skepticism.",
])
