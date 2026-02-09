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
st.set_page_config(page_title="Ch 54: Bayes' Theorem", layout="wide")
df = load_data()
fdf = sidebar_filters(df)

chapter_header(54, "Bayes' Theorem", part="XIII")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "Bayes' Theorem",
    "Bayes' theorem is what happens when you actually take the base rate seriously, "
    "which is something humans are famously terrible at. It tells us how to "
    "<b>update beliefs</b> when we observe new evidence -- a process that sounds trivial "
    "until you realize almost nobody does it correctly by default.<br><br>"
    "Given a hypothesis H and observed evidence E:<br><br>"
    "P(H | E) = P(E | H) * P(H) / P(E)<br><br>"
    "<b>Prior</b> P(H): what you believed before seeing any data. Your starting guess, "
    "which can be wrong and that is fine.<br>"
    "<b>Likelihood</b> P(E | H): if your hypothesis were true, how probable is the evidence "
    "you actually observed? This is the part that does the heavy lifting.<br>"
    "<b>Posterior</b> P(H | E): your updated belief after seeing the data. The whole point.<br>"
    "<b>Evidence</b> P(E): the total probability of observing the evidence across all hypotheses. "
    "Mostly serves as a normalising constant so the numbers add up to 1.",
)

formula_box(
    "Bayes' Theorem",
    r"P(H_i \mid E) = \frac{P(E \mid H_i) \, P(H_i)}{\sum_j P(E \mid H_j) \, P(H_j)}",
    "The denominator sums over all hypotheses to ensure the posterior probabilities add to 1. "
    "This is bookkeeping, not deep philosophy, but it is bookkeeping that matters.",
)

st.markdown("""
### Intuition with Weather Data

Here is a game. You receive a weather reading -- temperature and humidity -- but you
**don't know which city** it came from. Can you figure out the city using Bayes' theorem?

It turns out you can, and the reasoning is delightfully mechanical:

- **Hypothesis**: the reading came from city *i*
- **Evidence**: the observed temperature and humidity values
- **Prior**: your initial belief about which city (start with equal chances if you have no clue)
- **Likelihood**: how typical are these readings for each city, based on historical data?
- **Posterior**: your updated probability for each city after seeing the evidence

The beautiful thing is that this works even if your prior is terrible. But I am getting ahead of myself.
""")

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Step-by-Step Bayesian City Identification
# ---------------------------------------------------------------------------
st.subheader("Interactive: Identify the City from Weather Readings")

st.markdown(
    "Set the observed temperature and humidity below, and watch Bayes' theorem do its thing. "
    "We will compute P(city | features) step by step, so you can see exactly where the "
    "information comes from."
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
    f"Given temp={obs_temp} deg C and humidity={obs_hum}%, the most likely city is "
    f"**{most_likely}** with posterior probability {posteriors[most_likely]:.1%}. "
    "Notice how the posterior is not just the likelihood -- it is the likelihood "
    "weighted by the prior, then normalised. Every piece plays a role."
)

st.divider()

# ---------------------------------------------------------------------------
# 4. Sequential Updating
# ---------------------------------------------------------------------------
st.subheader("Sequential Bayesian Updating")

concept_box(
    "Sequential Updating",
    "Here is the genuinely magical thing about Bayesian inference: you can process evidence "
    "<b>one observation at a time</b>, and you get the same answer as if you processed it "
    "all at once. The posterior from observation #1 becomes the prior for observation #2, "
    "and so on. This is not an approximation. It is mathematically exact.<br><br>"
    "This means Bayesian reasoning is inherently <em>incremental</em>. You never have to "
    "start over. New data? Just update. It is the ideal framework for a world where "
    "information arrives continuously.",
)

st.markdown(
    "Watch how the posterior evolves as we observe temperature readings one by one "
    "from a secretly selected city. Notice how the model gets more and more confident "
    "as evidence accumulates -- or occasionally gets confused, which is also instructive."
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
        f"({current_prior[seq_city]:.1%}). Try adding more observations."
    )

insight_box(
    "Sequential updating is mathematically equivalent to batch updating -- you get the "
    "same answer either way. But watching it happen step by step reveals something important: "
    "the posterior converges to the true city as evidence accumulates, regardless of the prior. "
    "The data eventually speaks for itself."
)

st.divider()

# ---------------------------------------------------------------------------
# 5. Prior Sensitivity
# ---------------------------------------------------------------------------
st.subheader("Prior Sensitivity Analysis")

st.markdown(
    "You might ask: if the prior doesn't matter in the long run, why bother specifying one? "
    "Fair question. The answer is that it matters in the *short* run. With limited data, "
    "the prior and the likelihood arm-wrestle, and either can win. Let's compare three priors "
    "-- a uniform one, a correct one, and a deliberately wrong one -- and watch the data "
    "gradually overwhelm all of them."
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

insight_box(
    "Even with a wrong prior, the posterior converges to the truth as data accumulates. "
    "With enough evidence, the data overwhelms the prior -- this is called Bayesian consistency, "
    "and it is one of the genuinely reassuring results in all of statistics. "
    "You do not need to be right at the start. You just need to keep updating."
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
    explanation="The likelihood asks: 'If this hypothesis were true, how surprised would I be "
    "to see this evidence?' It is the engine that drives the update.",
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
    explanation="With enough data, the likelihood dominates the prior. This is called Bayesian "
    "consistency -- and it means you do not need to be right at the start, just willing to update.",
    key="ch54_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 8. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Bayes' theorem provides a principled, mechanical framework for updating beliefs with new evidence. It is not optional epistemology -- it is the only consistent way to do it.",
    "The prior reflects initial belief, the likelihood measures data compatibility, and the posterior is the updated belief. Each piece has a job.",
    "Sequential updating processes one observation at a time -- the result is mathematically identical to batch updating. You never need to start over.",
    "With enough data, the posterior is dominated by the likelihood, not the prior. Wrong priors are self-correcting.",
    "Bayesian reasoning naturally quantifies uncertainty, which is what makes it foundational to modern data science -- and, arguably, to thinking clearly about anything.",
])

navigation(
    prev_label="Ch 53: Bayesian Thinking",
    prev_page="53_Bayesian_Thinking.py",
    next_label="Ch 55: Bayesian Inference",
    next_page="55_Bayesian_Inference.py",
)
