"""Chapter 44: Bias-Variance Tradeoff -- Underfitting, overfitting, learning curves."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(44, "Bias-Variance Tradeoff", part="X")

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Setup — what are we even doing?
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "Let me set up a very specific problem, because I think the bias-variance tradeoff "
    "only makes sense when you can see it happening to a concrete prediction task."
)
st.markdown(
    "**The task**: You live in Dallas. It's day 180 of the year (late June). You want to "
    "predict the average temperature on that day. You have two years of historical hourly "
    "temperature readings -- roughly 17,500 data points for Dallas alone. Each data point "
    "is a row in our dataset: a timestamp, a day-of-year number (1 through 365), and the "
    "temperature in Celsius at that hour."
)
st.markdown(
    "You're going to build a model that takes **day-of-year** as input and outputs "
    "**predicted temperature**. That's it. One number in, one number out. The simplest "
    "possible prediction task. And yet this simple task is enough to illustrate the single "
    "most important idea in all of machine learning."
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Bias and variance WITHOUT jargon first
# ─────────────────────────────────────────────────────────────────────────────
st.header("Two Ways Your Prediction Can Go Wrong")

st.markdown(
    "Before I use the words 'bias' or 'variance,' I want you to see the two failure "
    "modes with your own eyes. They're genuinely different problems, and confusing them "
    "leads to wasting time on the wrong fix."
)

st.markdown("### Failure Mode 1: Your model is too simple for reality")
st.markdown(
    "Imagine I told you: 'Predict Dallas's temperature for every day of the year, but "
    "you're only allowed to use a straight line. One slope, one intercept, that's it.' "
    "You'd draw a line that maybe slopes gently upward or downward across the year. "
    "But Dallas's actual temperature pattern looks like a hill -- cold in January, hot "
    "in July, cold again in December. A straight line *cannot capture a hill*. It's not "
    "that you need more data, or better data, or a longer training run. The problem is "
    "structural: you've chosen a model shape that is incapable of representing the real "
    "pattern, no matter what."
)
st.markdown(
    "This is like trying to describe a circle using only straight lines. You can tilt "
    "your one straight line however you want -- it will never be a circle. Your predictions "
    "will be systematically wrong in a predictable way: too high in winter, too low in "
    "summer, wrong everywhere for the same underlying reason."
)

st.markdown("### Failure Mode 2: Your model memorizes noise instead of learning the pattern")
st.markdown(
    "Now imagine the opposite. I tell you: 'Predict Dallas's temperature, and you can "
    "use a degree-20 polynomial. Twenty coefficients. Go wild.' This polynomial is so "
    "flexible it can wiggle through almost every training data point. On day 47, the "
    "training data shows 8.3°C? The polynomial bends to hit 8.3. Day 48 shows 12.1°C? "
    "It bends again. Day 49 shows 7.9°C? Another wiggle."
)
st.markdown(
    "But here's the thing: the temperature on day 47 wasn't 8.3°C because of some deep "
    "pattern specific to day 47. It was 8.3°C because that's what happened to occur that "
    "year -- maybe a cold front came through, maybe it was cloudy. Next year, day 47 might "
    "be 11.5°C. The degree-20 polynomial doesn't know that. It has faithfully memorized "
    "that day 47 'should be' 8.3°C, because that's what the training data said. It has "
    "confused the *noise* (random weather fluctuation) for the *signal* (the seasonal "
    "pattern)."
)
st.markdown(
    "Your predictions will be great on the exact days you trained on. But ask it about "
    "a day it hasn't seen -- or next year's version of the same day -- and it'll give "
    "you something wild, because all those wiggles it learned were tracking randomness, "
    "not reality."
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: NOW introduce the terms
# ─────────────────────────────────────────────────────────────────────────────
st.header("Now Let's Name These Things")

st.markdown(
    "Failure Mode 1 -- 'your model is structurally too simple' -- is called **high bias**. "
    "The word 'bias' here means your model has a built-in tendency to be wrong in a "
    "specific direction. A straight line fitted to seasonal temperatures is *biased* "
    "toward flatness. It will always underestimate summer temps and overestimate winter "
    "temps, no matter how much data you throw at it. The error is consistent and "
    "predictable. If you re-ran the experiment with different training data from the same "
    "city, you'd get almost the same (wrong) straight line."
)

st.markdown(
    "Failure Mode 2 -- 'your model memorizes noise' -- is called **high variance**. "
    "The word 'variance' here means: if you trained the same type of model on *different* "
    "random samples of Dallas temperature data, you'd get wildly different predictions. "
    "Monday's sample might produce a polynomial that zigs where Tuesday's zags. The model "
    "is unstable -- it's not converging on the truth, it's chasing whatever noise happened "
    "to be in the particular sample it saw."
)

st.markdown(
    "Here's a useful way to think about it. Take 10 different random samples of Dallas "
    "temperature data. Train your model on each sample separately. Now look at the 10 "
    "predictions for, say, day 180:"
)

col1, col2 = st.columns(2)
with col1:
    concept_box(
        "High Bias (too simple)",
        "All 10 models predict roughly the same thing for day 180 -- say, 22°C, 22.1°C, "
        "21.9°C. They're consistent with each other. <b>But they're all wrong.</b> "
        "The real average is 33°C. They agree on the wrong answer because the model "
        "structure itself is flawed. Consistency without accuracy."
    )
with col2:
    concept_box(
        "High Variance (too complex)",
        "The 10 models predict all over the map: 28°C, 37°C, 19°C, 41°C, 25°C... "
        "Their <em>average</em> might actually be close to the true 33°C! But any "
        "individual model is unreliable. They can't agree because each one memorized "
        "different noise. Accuracy on average, but no single model you can trust."
    )

st.markdown(
    "And then there's a third source of error that has nothing to do with your model: "
    "**irreducible noise**. Even if you had a perfect model of Dallas's seasonal "
    "temperature pattern, you still couldn't predict the *exact* temperature on any given "
    "day 180, because weather has genuine randomness. Cold fronts, cloud cover, random "
    "atmospheric turbulence -- these create day-to-day variation that no model based on "
    "day-of-year alone can capture. This isn't a failure of modeling. It's a fact about "
    "the world."
)

formula_box(
    "The Decomposition",
    r"\mathbb{E}\!\left[(y - \hat{f}(x))^2\right] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2",
    "Your prediction error on any new data point is exactly the sum of three things: "
    "(1) how far off your model is on average (bias squared), (2) how much your model "
    "bounces around depending on which training data it saw (variance), and (3) the "
    "inherent unpredictability of the thing you're trying to predict (sigma squared, "
    "the irreducible noise). You can't touch sigma. Your job is to minimize bias + variance.",
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Interactive polynomial playground
# ─────────────────────────────────────────────────────────────────────────────
st.header("See It Yourself: Fitting the Annual Temperature Curve")

st.markdown(
    "Here's the actual experiment. We take a city's temperature data, compute the "
    "average temperature for each day of the year (so we have 365 data points: "
    "day 1 → avg temp, day 2 → avg temp, ... day 365 → avg temp). We hold out "
    "30% of these days as a test set the model never sees during training. Then we "
    "fit a polynomial of degree *d* to the training days and see how well it predicts "
    "the held-out test days."
)
st.markdown(
    "Degree 1 = straight line (almost certainly too simple). "
    "Degree 4 = a curve that can rise, peak, fall, and trough (probably about right "
    "for a seasonal pattern). Degree 20 = a wiggly monster with 20 bending points "
    "(almost certainly too complex). **Drag the slider and watch what happens.**"
)

poly_city = st.selectbox("City", CITY_LIST, key="poly_city")
city_annual = fdf[fdf["city"] == poly_city].groupby("day_of_year")["temperature_c"].mean().reset_index()
city_annual.columns = ["day_of_year", "avg_temp"]

poly_degree = st.slider("Polynomial degree (complexity knob)", 1, 20, 3, key="poly_deg")

X_day = city_annual["day_of_year"].values.reshape(-1, 1)
y_temp = city_annual["avg_temp"].values

# Train-test split for the annual curve
X_tr, X_te, y_tr, y_te = train_test_split(X_day, y_temp, test_size=0.3, random_state=42)

poly_model = make_pipeline(PolynomialFeatures(poly_degree), LinearRegression())
poly_model.fit(X_tr, y_tr)

train_rmse = np.sqrt(mean_squared_error(y_tr, poly_model.predict(X_tr)))
test_rmse = np.sqrt(mean_squared_error(y_te, poly_model.predict(X_te)))

# Generate smooth prediction line
X_smooth = np.linspace(1, 365, 365).reshape(-1, 1)
y_smooth = poly_model.predict(X_smooth)

fig_poly = go.Figure()
fig_poly.add_trace(go.Scatter(
    x=X_tr.flatten(), y=y_tr,
    mode="markers", marker=dict(size=5, color="#2A9D8F", opacity=0.6),
    name="Training days (model sees these)",
))
fig_poly.add_trace(go.Scatter(
    x=X_te.flatten(), y=y_te,
    mode="markers", marker=dict(size=5, color="#F4A261", opacity=0.6, symbol="diamond"),
    name="Test days (model has NEVER seen these)",
))
fig_poly.add_trace(go.Scatter(
    x=X_smooth.flatten(), y=y_smooth,
    mode="lines", line=dict(color="#E63946", width=3),
    name=f"Degree {poly_degree} polynomial prediction",
))
apply_common_layout(fig_poly, title=f"Day of Year → Temperature: {poly_city} (degree {poly_degree})", height=500)
fig_poly.update_layout(xaxis_title="Day of Year (1 = Jan 1, 180 = late June, 365 = Dec 31)",
                       yaxis_title="Average Temperature (°C)")
st.plotly_chart(fig_poly, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Train RMSE", f"{train_rmse:.2f}°C",
          help="How far off the model is on days it trained on")
c2.metric("Test RMSE", f"{test_rmse:.2f}°C",
          help="How far off the model is on days it has NEVER seen")
c3.metric("Gap (test − train)", f"{test_rmse - train_rmse:.2f}°C",
          help="A large gap means overfitting: great on training data, bad on new data")

if poly_degree == 1:
    st.info(
        f"**Degree 1 is a straight line.** Look at the plot -- it can't bend at all. "
        f"Dallas (or whichever city you picked) gets hot in summer and cold in winter, "
        f"but a straight line just cuts through the middle, wrong everywhere. "
        f"Train RMSE = {train_rmse:.2f}°C, test RMSE = {test_rmse:.2f}°C -- both are "
        f"bad, and they're bad by roughly the same amount. That's the fingerprint of "
        f"high bias: consistently wrong, but consistently wrong in the same way."
    )
elif poly_degree == 2:
    st.info(
        f"**Degree 2 is a parabola.** It can make one hump, which is better than a "
        f"straight line for a seasonal curve, but a parabola is symmetric and seasons "
        f"aren't perfectly symmetric (spring warms faster than fall cools, or vice versa). "
        f"Still underfitting, but less dramatically than degree 1."
    )
elif poly_degree <= 6:
    st.success(
        f"**Degree {poly_degree} is in the sweet spot.** The polynomial can bend enough to "
        f"follow the seasonal rise-peak-fall-trough pattern without wiggling wildly between "
        f"individual days. Train RMSE ({train_rmse:.2f}°C) and test RMSE ({test_rmse:.2f}°C) "
        f"are close to each other -- the model generalizes. It learned the *pattern* "
        f"(seasons exist) without memorizing the *noise* (what happened on any specific day)."
    )
elif poly_degree <= 12:
    st.warning(
        f"**Degree {poly_degree} is getting suspicious.** The curve is starting to wiggle "
        f"between data points. Train RMSE ({train_rmse:.2f}°C) looks great -- the model is "
        f"bending to fit the training points more closely. But test RMSE ({test_rmse:.2f}°C) "
        f"is higher. That growing gap between train and test is the early warning sign of "
        f"overfitting. The model is starting to learn that 'day 142 was 0.3°C warmer than "
        f"day 141' and treating that as a real pattern rather than noise."
    )
else:
    st.error(
        f"**Degree {poly_degree} has gone off the rails.** Look at the edges of the plot -- "
        f"the polynomial is probably shooting off to extreme values at the boundaries. "
        f"It has enough flexibility to thread through every training point, so train RMSE "
        f"is tiny ({train_rmse:.2f}°C). But test RMSE ({test_rmse:.2f}°C) has exploded "
        f"because the model has memorized the *specific noise* in the training data and "
        f"treats it as gospel. Every random fluctuation has been immortalized as a feature. "
        f"This is high variance: the model would look completely different if you trained "
        f"it on a different 70% of the data."
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: The U-shaped curve
# ─────────────────────────────────────────────────────────────────────────────
st.header("The Most Important Chart in Machine Learning")

st.markdown(
    "Now let's do what we just did by hand -- but systematically. Train a polynomial "
    "of every degree from 1 to 20, measure train RMSE and test RMSE for each, and plot "
    "them side by side."
)

degrees = list(range(1, 21))
train_errors, test_errors = [], []

for d in degrees:
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(X_tr, y_tr)
    train_errors.append(np.sqrt(mean_squared_error(y_tr, model.predict(X_tr))))
    test_errors.append(np.sqrt(mean_squared_error(y_te, model.predict(X_te))))

# Clip extreme test errors for readable y-axis
test_errors_clipped = np.clip(test_errors, 0, np.percentile(test_errors, 90) * 2)

fig_bv = go.Figure()
fig_bv.add_trace(go.Scatter(x=degrees, y=train_errors, mode="lines+markers",
                             name="Train RMSE (error on data model HAS seen)",
                             line=dict(color="#2A9D8F", width=2), marker=dict(size=6)))
fig_bv.add_trace(go.Scatter(x=degrees, y=test_errors_clipped, mode="lines+markers",
                             name="Test RMSE (error on data model has NEVER seen)",
                             line=dict(color="#E63946", width=2), marker=dict(size=6)))
fig_bv.add_vline(x=poly_degree, line_dash="dash", line_color="#F4A261",
                  annotation_text=f"Your choice: degree {poly_degree}")
apply_common_layout(fig_bv, title=f"Train vs Test Error by Polynomial Degree — {poly_city}", height=500)
fig_bv.update_layout(
    xaxis_title="Polynomial Degree (→ more complex)",
    yaxis_title="RMSE (°C) — lower is better",
    xaxis=dict(dtick=1),
)
st.plotly_chart(fig_bv, use_container_width=True)

st.markdown(
    "**Read this chart carefully.** The green line (training error) slopes steadily "
    "downward. Of course it does -- the more flexible the model, the more closely it "
    "can thread through the training points. If you give a polynomial enough degrees "
    "of freedom, it can eventually hit every single training point exactly, driving "
    "train RMSE to zero."
)
st.markdown(
    "The red line (test error) tells a completely different story. It drops at first -- "
    "going from degree 1 to degree 3 or 4 genuinely helps, because the model is learning "
    "the real seasonal pattern. But then it bottoms out and starts climbing. Why? Because "
    "every additional degree of flexibility beyond what the data actually requires gets "
    "used to memorize noise. And noise doesn't generalize."
)

best_idx = np.argmin(test_errors)
st.markdown(
    f"**The bottom of the U is at degree {degrees[best_idx]}** (test RMSE = "
    f"{test_errors[best_idx]:.2f}°C). Everything to the left of that point is underfitting: "
    f"the model is too simple and is leaving real signal on the table. Everything to the "
    f"right is overfitting: the model is too complex and is mistaking noise for signal."
)

insight_box(
    "This U-shaped curve isn't specific to polynomials or weather data. It shows up "
    "everywhere in machine learning. Decision tree depth, neural network size, number "
    "of features, regularization strength -- any knob that controls model complexity "
    "produces the same shape. The bottom of the U is always where you want to be."
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: The three-way comparison
# ─────────────────────────────────────────────────────────────────────────────
st.header("Side by Side: Underfitting vs Just Right vs Overfitting")

st.markdown(
    "One more way to see it. Here are three polynomial fits on the same data, "
    "all at once: degree 1 (too simple), degree 4 (about right), and degree 18 "
    "(way too complex). The gray dots are test data -- days the model never saw."
)

fig_cmp = go.Figure()
fig_cmp.add_trace(go.Scatter(
    x=X_te.flatten(), y=y_te,
    mode="markers", name="Test data (never seen by model)",
    marker=dict(color="gray", size=4, opacity=0.5),
))

cmp_configs = [
    (1, "#264653", "dot", "Degree 1 (high bias / underfitting)"),
    (4, "#2A9D8F", "solid", "Degree 4 (sweet spot)"),
    (18, "#E63946", "dash", "Degree 18 (high variance / overfitting)"),
]
for d, color, dash, label in cmp_configs:
    pm = make_pipeline(PolynomialFeatures(d), LinearRegression())
    pm.fit(X_tr, y_tr)
    y_c = pm.predict(X_smooth)
    te_rmse = np.sqrt(mean_squared_error(y_te, pm.predict(X_te)))
    fig_cmp.add_trace(go.Scatter(
        x=X_smooth.flatten(), y=y_c,
        mode="lines", name=f"{label} — test RMSE={te_rmse:.1f}°C",
        line=dict(color=color, width=2.5, dash=dash),
    ))

apply_common_layout(fig_cmp, title=f"Three Models, One Dataset — {poly_city}", height=500)
fig_cmp.update_layout(
    xaxis_title="Day of Year",
    yaxis_title="Temperature (°C)",
    yaxis_range=[city_annual["avg_temp"].min() - 10, city_annual["avg_temp"].max() + 10],
)
st.plotly_chart(fig_cmp, use_container_width=True)

st.markdown(
    "The blue dotted line (degree 1) cuts straight through everything -- it 'knows' "
    "the average temperature but has no concept of seasons. The green solid line "
    "(degree 4) follows the seasonal arc beautifully. The red dashed line (degree 18) "
    "probably looks okay in the middle but goes berserk near the edges, producing "
    "predictions like -50°C in January or 80°C in December."
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: A different task — city classification
# ─────────────────────────────────────────────────────────────────────────────
st.header("Same Idea, Different Task: Predicting Which City")

st.markdown(
    "The bias-variance tradeoff isn't just about polynomial curves. It applies to "
    "*every* model on *every* task. Let me show you the same phenomenon on a "
    "completely different problem."
)
st.markdown(
    "**New task**: Given a single weather reading -- temperature, humidity, wind speed, "
    "and pressure -- predict which of our 6 cities it came from. We'll use a random "
    "forest classifier, and the complexity knob is `max_depth`: how many questions "
    "each decision tree is allowed to ask."
)
st.markdown(
    "- `max_depth=1`: Each tree asks ONE question. 'Is temperature above 20°C?' That's "
    "it. Obviously too crude to distinguish 6 cities.\n"
    "- `max_depth=5`: Each tree asks up to 5 questions. 'Is temperature above 20°C? If yes, "
    "is humidity below 60%? If yes, is wind speed above 15 km/h?...' Starting to get useful.\n"
    "- `max_depth=30`: Each tree can ask 30 questions. It can create rules like 'if temp is "
    "between 22.7 and 22.8°C AND humidity is between 71.2 and 71.3% AND...' -- rules so "
    "specific they only match one or two training examples. That's memorization, not learning."
)

st.subheader("Learning Curves: Does More Data Help?")

st.markdown(
    "A **learning curve** plots accuracy vs training set size. It answers a crucial "
    "practical question: 'I have mediocre results. Should I go collect more data, or "
    "should I try a different model?' The answer depends on whether you're in high-bias "
    "or high-variance territory."
)

lc_depth = st.slider(
    "Random Forest max_depth", 1, 30, 10, key="lc_depth",
    help="Higher = more complex trees that can memorize more specific patterns"
)

le = LabelEncoder()
X_lc = fdf[FEATURE_COLS].dropna()
y_lc = le.fit_transform(fdf.loc[X_lc.index, "city"])

# Subsample for speed
sample_n = min(6000, len(X_lc))
rng = np.random.RandomState(42)
idx = rng.choice(len(X_lc), sample_n, replace=False)
X_lc_s = X_lc.iloc[idx].values
y_lc_s = y_lc[idx]

rf_lc = RandomForestClassifier(n_estimators=50, max_depth=lc_depth, random_state=42, n_jobs=-1)

with st.spinner("Computing learning curves (training the model at 10 different dataset sizes, 5-fold CV each)..."):
    train_sizes_abs, train_scores, val_scores = learning_curve(
        rf_lc, X_lc_s, y_lc_s,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring="accuracy", n_jobs=-1,
    )

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

fig_lc = go.Figure()
fig_lc.add_trace(go.Scatter(
    x=train_sizes_abs, y=train_mean,
    mode="lines+markers", name="Training accuracy",
    line=dict(color="#2A9D8F"),
    error_y=dict(type="data", array=train_std, visible=True),
))
fig_lc.add_trace(go.Scatter(
    x=train_sizes_abs, y=val_mean,
    mode="lines+markers", name="Validation accuracy (on held-out fold)",
    line=dict(color="#E63946"),
    error_y=dict(type="data", array=val_std, visible=True),
))
apply_common_layout(fig_lc, title=f"Learning Curve: City Classification (max_depth={lc_depth})", height=450)
fig_lc.update_layout(xaxis_title="Number of training examples", yaxis_title="Accuracy (higher is better)")
st.plotly_chart(fig_lc, use_container_width=True)

gap = train_mean[-1] - val_mean[-1]
if lc_depth <= 2:
    st.info(
        f"**High bias / underfitting.** With max_depth={lc_depth}, the trees can only ask "
        f"{lc_depth} question(s). Training accuracy = {train_mean[-1]:.3f}, validation accuracy "
        f"= {val_mean[-1]:.3f}. Both are low, and they're close together. More data won't "
        f"help here -- the model has already plateaued. It's not that it hasn't seen enough "
        f"examples; it's that it can't *represent* the decision boundary between cities with "
        f"only {lc_depth} split(s). **The fix: increase complexity** (raise max_depth)."
    )
elif gap > 0.1:
    st.warning(
        f"**High variance / overfitting.** Training accuracy = {train_mean[-1]:.3f}, but "
        f"validation accuracy = {val_mean[-1]:.3f} -- a gap of {gap:.3f}. The model is "
        f"memorizing the training data (it recognizes specific weather readings it's seen "
        f"before) but fails on new readings. Notice the validation curve is still rising "
        f"with more data -- that's the signature of high variance. **The fix: get more data**, "
        f"or reduce complexity (lower max_depth), or use regularization."
    )
else:
    st.success(
        f"**Good balance.** Training = {train_mean[-1]:.3f}, validation = {val_mean[-1]:.3f} "
        f"(gap = {gap:.3f}). The model is complex enough to capture the weather differences "
        f"between cities but not so complex that it memorizes individual readings."
    )

# Compare different complexities
st.subheader("What Does 'Try a Different Complexity' Look Like?")
st.markdown(
    "Here are learning curves at four different tree depths, side by side. Watch how "
    "the curve shape changes:"
)

depths_to_compare = [1, 5, 15, None]
fig_multi = go.Figure()
colors = ["#7209B7", "#FB8500", "#2A9D8F", "#E63946"]

for depth, color in zip(depths_to_compare, colors):
    rf_temp = RandomForestClassifier(n_estimators=50, max_depth=depth, random_state=42, n_jobs=-1)
    ts, t_sc, v_sc = learning_curve(
        rf_temp, X_lc_s, y_lc_s,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=3, scoring="accuracy", n_jobs=-1,
    )
    label = f"depth={depth}" if depth else "depth=unlimited"
    fig_multi.add_trace(go.Scatter(
        x=ts, y=v_sc.mean(axis=1),
        mode="lines+markers", name=label,
        line=dict(color=color, width=2),
    ))

apply_common_layout(fig_multi, title="Validation Accuracy at Different Tree Depths", height=450)
fig_multi.update_layout(xaxis_title="Training Set Size", yaxis_title="Validation Accuracy")
st.plotly_chart(fig_multi, use_container_width=True)

insight_box(
    "Depth=1 (purple) flatlines around low accuracy -- it's hit the ceiling of what "
    "one question can do. No amount of additional weather data helps because the model "
    "literally can't represent the answer. Depth=unlimited (red) starts lower on small "
    "datasets (overfitting to tiny samples) but catches up as data grows. The moderate "
    "depths (orange, green) tend to perform best across the range. The lesson: when your "
    "model is too simple, get a better model. When your model is too complex, get more data "
    "(or simplify)."
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Code
# ─────────────────────────────────────────────────────────────────────────────
code_example("""from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Prepare city classification data
X = df[['temperature_c', 'relative_humidity_pct', 'wind_speed_kmh', 'surface_pressure_hpa']].values
y = df['city'].values  # 6 cities

model = RandomForestClassifier(max_depth=10, n_estimators=50)

# Compute learning curves: train model at 10 different dataset sizes, 5-fold CV each
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy'
)

# High bias: both curves plateau low, close together -> need more complex model
# High variance: big gap between curves -> need more data or simpler model
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()

quiz(
    "You fit a straight line to Dallas's annual temperature pattern. Train RMSE is 8.1°C, "
    "test RMSE is 8.3°C. What's the problem?",
    [
        "Overfitting — the model memorized the training data",
        "Underfitting — the model is too simple to capture the seasonal curve",
        "The data is too noisy to model",
        "The test set is too small",
    ],
    correct_idx=1,
    explanation=(
        "Both train and test RMSE are high AND they're nearly equal. That's the "
        "fingerprint of high bias (underfitting). The straight line is consistently "
        "wrong because it structurally can't represent a seasonal curve. A more "
        "flexible model (higher polynomial degree) would help. More data would not."
    ),
    key="ch44_quiz1",
)

quiz(
    "You fit a degree-15 polynomial. Train RMSE is 0.2°C, test RMSE is 14.7°C. What's happening?",
    [
        "Underfitting — the model needs more flexibility",
        "Overfitting — the model memorized noise in the training data",
        "The seasonal pattern doesn't exist",
        "The test data is from a different city",
    ],
    correct_idx=1,
    explanation=(
        "Train RMSE of 0.2°C means the polynomial hits almost every training point. "
        "But test RMSE of 14.7°C means it's wildly wrong on new data. That gap -- "
        "0.2 vs 14.7 -- is the textbook signature of overfitting (high variance). "
        "The polynomial used all its flexibility to memorize the specific noise in "
        "the training data, so when it encounters different noise in the test data, "
        "it flails. The fix: reduce the polynomial degree, or add regularization."
    ),
    key="ch44_quiz2",
)

quiz(
    "Your city classification model has training accuracy 72% and validation accuracy 71%. "
    "What should you do?",
    [
        "Collect more data — the model needs more examples",
        "Use a more complex model — this one is too simple (high bias)",
        "Reduce model complexity — it's overfitting",
        "Nothing — 72% is probably fine",
    ],
    correct_idx=1,
    explanation=(
        "Training and validation are nearly identical (gap of only 1%), so there's "
        "almost no overfitting. The problem is that both are only 72% -- the model "
        "has converged but converged to a mediocre answer. This is high bias. The "
        "model is too simple to distinguish all 6 cities well. More data won't help "
        "(the curves have already plateaued). A more complex model -- deeper trees, "
        "more features, a different algorithm -- is the right move."
    ),
    key="ch44_quiz3",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "**Bias** means your model is structurally too simple. A straight line can't capture "
    "Dallas's seasonal temperature curve no matter how much data you give it. The predictions "
    "are consistently wrong in a predictable direction.",
    "**Variance** means your model is too sensitive to the specific training data it saw. "
    "A degree-20 polynomial memorizes random weather fluctuations and produces wildly "
    "different curves depending on which days happened to be in the training set.",
    "**Irreducible noise** is the day-to-day weather randomness that no model based on "
    "day-of-year alone can capture. It sets a floor on your error that you cannot beat.",
    "The **U-shaped test error curve** is your guide: training error always decreases "
    "with complexity, but test error drops then rises. The bottom of the U is where "
    "bias and variance are balanced.",
    "**Learning curves** diagnose which problem you have: if train and validation are "
    "both low and close together, you need a more complex model. If there's a big gap, "
    "you need more data or a simpler model.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 43: Cross-Validation",
    prev_page="43_Cross_Validation.py",
    next_label="Ch 45: ROC/AUC & Confusion Matrix",
    next_page="45_ROC_AUC_Confusion_Matrix.py",
)
