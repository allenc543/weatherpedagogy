"""Chapter 54: Bayes' Theorem — Prior, likelihood, posterior, sequential updating."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)
from utils.constants import CITY_LIST, CITY_COLORS, FEATURE_COLS, FEATURE_LABELS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
df = load_data()
fdf = sidebar_filters(df)

chapter_header(54, "Bayes' Theorem", part="XIII")

st.markdown(
    "Let me start with a concrete puzzle, because Bayes' theorem is one of those ideas "
    "that is simultaneously trivial and profound, and the only way to see why is to work "
    "through a specific example."
)
st.markdown(
    "**The puzzle**: Your friend hands you a weather reading -- temperature 35.0 C and "
    "humidity 30%. They tell you it came from one of our 6 cities (Dallas, Houston, Austin, "
    "San Antonio, NYC, or LA) but will not tell you which one. Can you figure it out?"
)
st.markdown(
    "Your first instinct might be: 'Just check which city's average temperature is "
    "closest to 35 C.' That is reasonable, but it ignores crucial information. What if "
    "35 C is common in both Dallas and Houston but the 30% humidity is very unusual for "
    "Houston (which averages ~65% humidity) and totally normal for Dallas in summer? "
    "The humidity narrows it down in a way that temperature alone cannot. Bayes' theorem "
    "gives us a *principled, mechanical* procedure for combining all this information."
)
st.markdown(
    "**Why this matters beyond a parlor game**: This is exactly the logic behind spam "
    "filters, medical diagnosis, weather model calibration, and any system that needs to "
    "update its beliefs based on evidence. The same machinery that identifies cities from "
    "weather readings can tell you whether a patient has a disease given a test result, "
    "or whether a forecast model should be trusted given recent observations."
)

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
st.markdown("### The Mechanics, Step by Step")

st.markdown(
    "Let me walk through the calculation for our specific puzzle before introducing any "
    "formal notation."
)
st.markdown(
    "**Step 1 -- Prior belief**: Before looking at the numbers, what do you believe? If "
    "you have no idea which city it came from, you give each city equal probability: "
    "1/6 = 16.7%. This is your **prior**."
)
st.markdown(
    "**Step 2 -- Check the evidence against each city**: Now look at the data: 35 C, 30% "
    "humidity. For each city, ask: 'How typical is this reading?' Dallas averages ~20 C "
    "with std ~10 C, so 35 C is 1.5 standard deviations above average -- unusual but "
    "not crazy. NYC averages ~13 C with std ~10 C, so 35 C is 2.2 standard deviations "
    "above -- very unusual. This 'how typical is the evidence for each city' is the "
    "**likelihood**."
)
st.markdown(
    "**Step 3 -- Combine**: Multiply each city's prior by its likelihood. Dallas might "
    "get 0.167 * 0.0032 = 0.00053. NYC might get 0.167 * 0.00004 = 0.0000067. Dallas's "
    "number is 80 times bigger, so Dallas is 80 times more likely."
)
st.markdown(
    "**Step 4 -- Normalize**: Make the probabilities sum to 1 by dividing each city's "
    "score by the total. This gives you the **posterior** -- your updated belief about "
    "which city produced this reading."
)

concept_box(
    "Bayes' Theorem",
    "Now the formal version. Given a hypothesis H (the reading came from city i) and "
    "observed evidence E (temperature 35 C, humidity 30%):<br><br>"
    "P(H | E) = P(E | H) * P(H) / P(E)<br><br>"
    "<b>Prior</b> P(H): what you believed before seeing the data. Started at 1/6 for each "
    "city. Could be wrong, and that is fine -- Bayes' theorem is self-correcting.<br>"
    "<b>Likelihood</b> P(E | H): if this reading came from Dallas, how probable is 35 C "
    "and 30% humidity? This is the part that does the heavy lifting.<br>"
    "<b>Posterior</b> P(H | E): your updated belief. The answer to 'which city?' after "
    "incorporating the evidence.<br>"
    "<b>Evidence</b> P(E): the total probability of seeing 35 C, 30% humidity across all "
    "cities. This is just a normalizing constant so the posteriors sum to 1.",
)

formula_box(
    "Bayes' Theorem",
    r"\underbrace{P(H_i \mid E)}_{\text{posterior probability}} = \frac{\underbrace{P(E \mid H_i)}_{\text{likelihood of evidence}} \, \underbrace{P(H_i)}_{\text{prior belief}}}{\underbrace{\sum_j P(E \mid H_j) \, P(H_j)}_{\text{total evidence}}}",
    "The denominator sums over all hypotheses (all 6 cities) to ensure the posterior "
    "probabilities add to 1. It is bookkeeping, not deep philosophy, but it is bookkeeping "
    "that matters -- without it your probabilities would not be probabilities.",
)

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Step-by-Step Bayesian City Identification
# ---------------------------------------------------------------------------
st.subheader("Interactive: Identify the City from Weather Readings")

st.markdown(
    "Now try it yourself. Set the observed temperature and humidity below, and watch "
    "Bayes' theorem process the evidence step by step."
)
st.markdown(
    "**What the controls do:** The temperature and humidity sliders set the observed "
    "weather reading. The prior type controls your starting belief: 'uniform' means no "
    "prior opinion (equal probability for each city), 'proportional to data count' means "
    "you believe cities appear in proportion to how many readings they have in our dataset, "
    "and 'custom' lets you set any prior you want."
)
st.markdown(
    "**What to look for:** Watch how the posterior shifts as you move the sliders. "
    "At 35 C / 30% humidity, Dallas or San Antonio should dominate (hot, dry Texas). "
    "At 5 C / 80% humidity, NYC should light up (cold, humid winter). At 22 C / 45%, "
    "LA should score well (mild, moderate). Notice how the posterior is not just the "
    "likelihood -- the prior matters too, especially when the likelihoods are similar."
)

col_ctrl, col_viz = st.columns([1, 2])

with col_ctrl:
    obs_temp = st.slider(
        "Observed Temperature (deg C)", -10.0, 45.0, 35.0, 0.5, key="bt_temp"
    )
    obs_hum = st.slider(
        "Observed Humidity (%)", 0.0, 100.0, 30.0, 1.0, key="bt_hum"
    )
    st.markdown("---")
    st.markdown("**Prior Type**")
    prior_type = st.radio(
        "Choose prior distribution over cities:",
        ["Uniform (equal probability)", "Proportional to data count", "Custom"],
        key="bt_prior_type",
    )

# Compute per-city statistics from filtered data
city_stats = {}
for city in CITY_LIST:
    cdata = fdf[fdf["city"] == city]
    if len(cdata) < 10:
        continue
    city_stats[city] = {
        "count": len(cdata),
        "temp_mean": cdata["temperature_c"].mean(),
        "temp_std": cdata["temperature_c"].std(),
        "hum_mean": cdata["relative_humidity_pct"].mean(),
        "hum_std": cdata["relative_humidity_pct"].std(),
    }

active_cities = list(city_stats.keys())

if len(active_cities) < 2:
    st.warning("Please select at least 2 cities in the sidebar to use this demo.")
    st.stop()

# --- Priors ---
if prior_type == "Uniform (equal probability)":
    priors = {c: 1.0 / len(active_cities) for c in active_cities}
elif prior_type == "Proportional to data count":
    total = sum(city_stats[c]["count"] for c in active_cities)
    priors = {c: city_stats[c]["count"] / total for c in active_cities}
else:
    st.markdown("**Set custom priors** (will be normalised):")
    raw_priors = {}
    cols_prior = st.columns(min(len(active_cities), 3))
    for i, city in enumerate(active_cities):
        with cols_prior[i % len(cols_prior)]:
            raw_priors[city] = st.number_input(
                city, min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                key=f"bt_prior_{city}",
            )
    total_raw = sum(raw_priors.values())
    if total_raw == 0:
        total_raw = 1.0
    priors = {c: v / total_raw for c, v in raw_priors.items()}

# --- Likelihoods (Gaussian assumption per feature) ---
likelihoods = {}
for city in active_cities:
    s = city_stats[city]
    # P(temp | city) * P(hum | city) assuming independence
    p_temp = stats.norm.pdf(obs_temp, loc=s["temp_mean"], scale=max(s["temp_std"], 0.1))
    p_hum = stats.norm.pdf(obs_hum, loc=s["hum_mean"], scale=max(s["hum_std"], 0.1))
    likelihoods[city] = p_temp * p_hum

# --- Evidence (normalising constant) ---
evidence = sum(likelihoods[c] * priors[c] for c in active_cities)

# --- Posterior ---
posteriors = {}
for city in active_cities:
    if evidence > 0:
        posteriors[city] = (likelihoods[city] * priors[city]) / evidence
    else:
        posteriors[city] = priors[city]

# ---------------------------------------------------------------------------
# 3. Visualise all three stages
# ---------------------------------------------------------------------------
with col_viz:
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Prior P(city)", "Likelihood P(obs | city)", "Posterior P(city | obs)"],
        shared_yaxes=False,
    )

    colors = [CITY_COLORS.get(c, "#888888") for c in active_cities]

    # Prior bars
    fig.add_trace(
        go.Bar(
            x=active_cities, y=[priors[c] for c in active_cities],
            marker_color=colors, name="Prior", showlegend=False,
            text=[f"{priors[c]:.3f}" for c in active_cities], textposition="outside",
        ),
        row=1, col=1,
    )

    # Likelihood bars
    max_lik = max(likelihoods.values()) if max(likelihoods.values()) > 0 else 1
    norm_lik = {c: likelihoods[c] / max_lik for c in active_cities}
    fig.add_trace(
        go.Bar(
            x=active_cities, y=[norm_lik[c] for c in active_cities],
            marker_color=colors, name="Likelihood", showlegend=False,
            text=[f"{norm_lik[c]:.3f}" for c in active_cities], textposition="outside",
        ),
        row=1, col=2,
    )

    # Posterior bars
    fig.add_trace(
        go.Bar(
            x=active_cities, y=[posteriors[c] for c in active_cities],
            marker_color=colors, name="Posterior", showlegend=False,
            text=[f"{posteriors[c]:.3f}" for c in active_cities], textposition="outside",
        ),
        row=1, col=3,
    )

    fig.update_layout(template="plotly_white", height=450, margin=dict(t=60, b=40))
    fig.update_yaxes(range=[0, 1.15], row=1, col=1)
    fig.update_yaxes(range=[0, 1.15], row=1, col=2)
    fig.update_yaxes(range=[0, 1.15], row=1, col=3)
    st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "**Reading the three panels:** The left panel shows your prior belief (starting point). "
    "The middle panel shows the likelihood -- how compatible the observed reading is with "
    "each city's historical weather. The right panel shows the posterior -- your updated "
    "belief after combining prior and likelihood. The posterior is the *product* of the "
    "first two panels, normalized to sum to 1."
)

# Detail table
st.markdown("#### Calculation Details")
detail_rows = []
for city in active_cities:
    s = city_stats[city]
    detail_rows.append({
        "City": city,
        "Prior": f"{priors[city]:.4f}",
        "Temp Mean": f"{s['temp_mean']:.1f}",
        "Temp Std": f"{s['temp_std']:.1f}",
        "Hum Mean": f"{s['hum_mean']:.1f}",
        "Hum Std": f"{s['hum_std']:.1f}",
        "Likelihood": f"{likelihoods[city]:.6e}",
        "Prior x Likelihood": f"{priors[city] * likelihoods[city]:.6e}",
        "Posterior": f"{posteriors[city]:.4f}",
    })
detail_df = pd.DataFrame(detail_rows)
st.dataframe(detail_df, use_container_width=True, hide_index=True)

most_likely = max(posteriors, key=posteriors.get)
insight_box(
    f"Given temp = {obs_temp} C and humidity = {obs_hum}%, the most likely city is "
    f"**{most_likely}** with posterior probability {posteriors[most_likely]:.1%}. "
    f"Look at the detail table: {most_likely} has the highest likelihood (the observed "
    f"reading is most consistent with its historical weather patterns), and after "
    f"multiplying by the prior and normalizing, it dominates the posterior. Try moving "
    f"the temperature to 5 C and humidity to 80% -- NYC should jump to the top because "
    f"cold, humid readings are typical there and rare in Texas."
)

st.divider()

# ---------------------------------------------------------------------------
# 4. Sequential Updating
# ---------------------------------------------------------------------------
st.subheader("Sequential Bayesian Updating")

st.markdown(
    "Here is the genuinely magical property of Bayes' theorem that makes it so powerful "
    "in practice: you can process evidence **one observation at a time**, and the result "
    "is mathematically identical to processing it all at once."
)
st.markdown(
    "The posterior from observation #1 becomes the prior for observation #2. Then the "
    "posterior from observation #2 becomes the prior for observation #3. And so on. "
    "This is not an approximation. It is mathematically exact. And it means Bayesian "
    "reasoning is inherently *incremental* -- you never have to start over when new "
    "data arrives."
)

concept_box(
    "Sequential Updating",
    "Imagine you are trying to identify a mystery city from a stream of temperature "
    "readings arriving one per hour. After the first reading (22.5 C), you update your "
    "beliefs. Maybe Dallas and Houston are tied. After the second reading (23.1 C), "
    "you update again -- now Dallas pulls slightly ahead. After reading #10 (31.2 C), "
    "if the readings keep looking like Dallas's distribution, Dallas's probability is "
    "now 0.85 and climbing.<br><br>"
    "The math is simple: <b>posterior_1 becomes prior_2</b>. posterior_2 becomes "
    "prior_3. Each observation tightens your belief. Eventually, one city pulls so "
    "far ahead that you are effectively certain -- and you got there one observation "
    "at a time.",
)

st.markdown(
    "**What the controls do:** 'Secret city' is the city from which we draw temperature "
    "readings. 'Number of sequential observations' controls how many readings the model "
    "sees. **What to look for:** Watch the secret city's probability line climb toward 1.0 "
    "as evidence accumulates. How many observations does it take to be 90% confident? "
    "Cities with distinctive temperatures (NYC, LA) are identified faster than cities with "
    "similar temperatures (Dallas vs Houston)."
)

seq_city = st.selectbox("Secret city (for demonstration)", CITY_LIST, key="bt_seq_city")
n_obs = st.slider("Number of sequential observations", 1, 30, 10, key="bt_nobs")

# Sample observations from the secret city
seq_data = fdf[fdf["city"] == seq_city]["temperature_c"].dropna()
if len(seq_data) < n_obs:
    st.warning("Not enough data. Try fewer observations or a different filter.")
    st.stop()

rng = np.random.RandomState(42)
observations = rng.choice(seq_data.values, size=n_obs, replace=False)

# Run sequential updating
current_prior = {c: 1.0 / len(active_cities) for c in active_cities}
history = [{"obs": "Start", **{c: current_prior[c] for c in active_cities}}]

for i, obs_val in enumerate(observations):
    # Compute likelihoods for this single observation
    liks = {}
    for city in active_cities:
        s = city_stats[city]
        liks[city] = stats.norm.pdf(obs_val, loc=s["temp_mean"], scale=max(s["temp_std"], 0.1))

    ev = sum(liks[c] * current_prior[c] for c in active_cities)
    new_posterior = {}
    for city in active_cities:
        new_posterior[city] = (liks[city] * current_prior[city]) / ev if ev > 0 else current_prior[city]

    history.append({"obs": f"#{i+1}: {obs_val:.1f} C", **{c: new_posterior[c] for c in active_cities}})
    current_prior = new_posterior.copy()

hist_df = pd.DataFrame(history)

fig_seq = go.Figure()
for city in active_cities:
    fig_seq.add_trace(go.Scatter(
        x=hist_df["obs"], y=hist_df[city],
        mode="lines+markers", name=city,
        line=dict(color=CITY_COLORS.get(city, "#888888"), width=2),
    ))
fig_seq.update_layout(
    yaxis_title="Posterior Probability", xaxis_title="Observation",
    legend_title="City",
)
apply_common_layout(fig_seq, title="Sequential Bayesian Updating of City Probabilities", height=500)
st.plotly_chart(fig_seq, use_container_width=True)

final_guess = max(current_prior, key=current_prior.get)
if final_guess == seq_city:
    st.success(
        f"After {n_obs} observations, the model correctly identifies **{seq_city}** "
        f"with probability {current_prior[seq_city]:.1%}!"
    )
else:
    st.warning(
        f"After {n_obs} observations, the model guesses **{final_guess}** "
        f"({current_prior[final_guess]:.1%}) but the true city is **{seq_city}** "
        f"({current_prior[seq_city]:.1%}). This happens when the secret city's temperatures "
        f"are similar to another city's. Try adding more observations -- the correct city "
        f"will eventually win as evidence accumulates."
    )

insight_box(
    "Sequential updating is mathematically equivalent to batch updating -- you get the "
    "same answer either way. But watching it step by step reveals something important: "
    "the speed of convergence depends on how distinctive the city's weather is. NYC "
    "(mean ~13 C) is identified quickly because its temperatures are very different from "
    "the Texas cities (~20 C). Dallas and Houston take more observations to distinguish "
    "because their temperature distributions overlap heavily. This is Bayes' theorem "
    "telling you: 'I need more evidence when the hypotheses are similar.'"
)

st.divider()

# ---------------------------------------------------------------------------
# 5. Prior Sensitivity
# ---------------------------------------------------------------------------
st.subheader("Prior Sensitivity Analysis")

st.markdown(
    "A common objection to Bayesian reasoning: 'But the prior is subjective! What if I "
    "pick the wrong one?' This is a legitimate concern, and the answer is both reassuring "
    "and quantifiable. Let us test it."
)
st.markdown(
    "We take 5 temperature readings from a target city and run Bayesian updating with "
    "three different priors: (1) uniform (no opinion), (2) correct (slightly favor the "
    "true city), and (3) deliberately wrong (favor a different city). Then we watch all "
    "three converge to the same answer."
)

target_city = st.selectbox("Target city for sensitivity analysis", active_cities, key="bt_sens_city")
other_city = [c for c in active_cities if c != target_city][0]

# Three different priors
prior_scenarios = {
    "Uniform": {c: 1.0 / len(active_cities) for c in active_cities},
    "Correct (favour true city)": {c: (0.6 if c == target_city else 0.4 / (len(active_cities) - 1))
                                    for c in active_cities},
    "Wrong (favour other city)": {c: (0.6 if c == other_city else 0.4 / (len(active_cities) - 1))
                                   for c in active_cities},
}

# Draw 5 observations from target city
sens_data = fdf[fdf["city"] == target_city]["temperature_c"].dropna()
if len(sens_data) < 5:
    st.info("Not enough data for this city.")
    st.stop()
sens_obs = rng.choice(sens_data.values, size=5, replace=False)

fig_sens = make_subplots(
    rows=1, cols=3,
    subplot_titles=list(prior_scenarios.keys()),
    shared_yaxes=True,
)

for col_idx, (scenario_name, prior) in enumerate(prior_scenarios.items(), 1):
    curr = prior.copy()
    steps = [curr[target_city]]
    for obs_val in sens_obs:
        liks = {}
        for city in active_cities:
            s = city_stats[city]
            liks[city] = stats.norm.pdf(obs_val, loc=s["temp_mean"], scale=max(s["temp_std"], 0.1))
        ev = sum(liks[c] * curr[c] for c in active_cities)
        for city in active_cities:
            curr[city] = (liks[city] * curr[city]) / ev if ev > 0 else curr[city]
        steps.append(curr[target_city])

    fig_sens.add_trace(
        go.Scatter(
            x=list(range(len(steps))), y=steps,
            mode="lines+markers",
            line=dict(color=CITY_COLORS.get(target_city, "#888"), width=2),
            showlegend=(col_idx == 1),
            name=f"P({target_city})",
        ),
        row=1, col=col_idx,
    )
    fig_sens.update_xaxes(title_text="Observation #", row=1, col=col_idx)

fig_sens.update_yaxes(title_text=f"P({target_city})", range=[0, 1.05], row=1, col=1)
fig_sens.update_layout(template="plotly_white", height=400, margin=dict(t=60, b=40))
st.plotly_chart(fig_sens, use_container_width=True)

st.markdown(
    f"**Reading the three panels:** Each panel tracks the probability of the true city "
    f"({target_city}) over 5 observations, starting from a different prior. The left panel "
    f"starts at ~17% (uniform), the middle at 60% (correct), the right at a low value "
    f"(wrong prior favoring {other_city}). Despite starting in very different places, "
    f"all three converge toward the same value by observation 5."
)

insight_box(
    "Even with a wrong prior, the posterior converges to the truth as data accumulates. "
    "This is called **Bayesian consistency**, and it is one of the genuinely reassuring "
    "results in all of statistics. The wrong prior starts you in the wrong place, but "
    "it does not keep you there. Five temperature readings are enough to mostly erase "
    "a bad prior. Fifty readings would erase it completely. You do not need to be right "
    "at the start. You just need to keep updating."
)

st.divider()

# ---------------------------------------------------------------------------
# 6. Code Example
# ---------------------------------------------------------------------------
code_example("""
import numpy as np
from scipy import stats

# City temperature statistics (mean, std)
city_params = {
    'Dallas': (20.5, 10.2),
    'Houston': (22.1, 8.5),
    'NYC': (13.8, 10.6),
}
cities = list(city_params.keys())

# Observed temperature
obs_temp = 35.0

# Step 1: Prior (uniform)
prior = {c: 1/len(cities) for c in cities}

# Step 2: Likelihood P(obs | city)
likelihood = {}
for city in cities:
    mu, sigma = city_params[city]
    likelihood[city] = stats.norm.pdf(obs_temp, loc=mu, scale=sigma)

# Step 3: Evidence (normalising constant)
evidence = sum(likelihood[c] * prior[c] for c in cities)

# Step 4: Posterior
posterior = {c: likelihood[c] * prior[c] / evidence for c in cities}
print("Posterior:", {c: f"{p:.3f}" for c, p in posterior.items()})
""")

st.divider()

# ---------------------------------------------------------------------------
# 7. Quiz
# ---------------------------------------------------------------------------
quiz(
    "In Bayes' theorem, what does the likelihood P(E|H) represent?",
    [
        "The probability of the hypothesis being true",
        "The probability of observing the evidence if the hypothesis is true",
        "The updated probability after seeing evidence",
        "The total probability of the evidence",
    ],
    correct_idx=1,
    explanation=(
        "The likelihood asks: 'If this reading really came from Dallas, how probable is a "
        "temperature of 35 C and humidity of 30%?' It checks the evidence against each "
        "hypothesis. Dallas might have a likelihood of 0.003 (35 C is plausible there), "
        "while NYC has a likelihood of 0.00004 (35 C is a record-breaking heat wave in NYC). "
        "The likelihood is the engine that drives the Bayesian update -- it is what converts "
        "data into evidence. A common confusion: the likelihood is NOT the probability of the "
        "hypothesis. P(E|H) and P(H|E) are very different things, and confusing them is called "
        "the prosecutor's fallacy."
    ),
    key="ch54_quiz1",
)

quiz(
    "If you start with a very wrong prior but observe 100 data points, what happens?",
    [
        "The posterior remains dominated by the wrong prior",
        "The posterior becomes undefined",
        "The data overwhelms the prior and the posterior converges to the truth",
        "You must restart with the correct prior",
    ],
    correct_idx=2,
    explanation=(
        "With enough data, the likelihood dominates the prior. This is Bayesian consistency, "
        "and you can see it directly in the prior sensitivity demo above. After just 5 "
        "temperature readings, even a deliberately wrong prior (60% probability on the wrong "
        "city) gets corrected. After 100 readings, the prior contributes effectively nothing "
        "to the posterior -- the data speaks for itself. You do not need to be right at the "
        "start. You just need to keep updating. This is why Bayesian methods are practical "
        "even when the prior is somewhat arbitrary."
    ),
    key="ch54_quiz2",
)

quiz(
    "You observe a temperature of 22 C. Dallas (mean 20.5 C, std 10.2 C) and Houston "
    "(mean 22.1 C, std 8.5 C) both have similar likelihoods. What does Bayes' theorem "
    "tell you to do?",
    [
        "Pick the city with the higher likelihood and ignore the prior",
        "The prior breaks the tie -- if you have reason to believe one city is more likely, that matters",
        "The theorem gives no answer when likelihoods are similar",
        "You need to use a different method entirely",
    ],
    correct_idx=1,
    explanation=(
        "When the likelihoods are similar, the prior matters a lot. This is exactly the "
        "situation where Bayesian reasoning shines: it gives you a principled way to break "
        "ties. If you know the reading came from Texas (maybe you know the sensor is in "
        "Texas but not which city), you might have a higher prior for Dallas than Houston. "
        "That prior tips the balance. In the sequential updating demo, this is why Dallas "
        "and Houston take many observations to distinguish -- their temperature distributions "
        "overlap heavily, so each individual observation provides only a little evidence. "
        "The prior decides the initial lean, and the data slowly accumulates enough evidence "
        "to override it."
    ),
    key="ch54_quiz3",
)

st.divider()

# ---------------------------------------------------------------------------
# 8. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Bayes' theorem provides a mechanical, principled procedure for updating beliefs with "
    "evidence. Given a weather reading (35 C, 30% humidity), it combines your prior belief "
    "about which city with how typical that reading is for each city to produce an updated "
    "probability.",
    "The **prior** is your starting belief, the **likelihood** measures how consistent the "
    "data is with each hypothesis, and the **posterior** is the updated belief. Each piece "
    "has a specific job in the calculation.",
    "**Sequential updating** processes one observation at a time: the posterior from reading "
    "#1 becomes the prior for reading #2. The result is mathematically identical to processing "
    "all readings at once. You never need to start over.",
    "With enough data, the posterior is dominated by the likelihood, not the prior. **Wrong "
    "priors are self-correcting** -- 5-10 observations are often enough to erase a bad prior. "
    "This is Bayesian consistency.",
    "Cities with distinctive weather (NYC: cold, humid) are identified faster than cities "
    "with similar weather (Dallas vs Houston: both hot Texas cities). Bayes' theorem "
    "automatically 'knows' when it needs more evidence.",
])

navigation(
    prev_label="Ch 53: Bayesian Thinking",
    prev_page="53_Bayesian_Thinking.py",
    next_label="Ch 55: Bayesian Inference",
    next_page="55_Bayesian_Inference.py",
)
