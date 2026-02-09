"""Chapter 55: Bayesian Inference — Parameter estimation, credible intervals, prior sensitivity."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.stats_helpers import bootstrap_ci
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)
from utils.constants import CITY_LIST, CITY_COLORS, FEATURE_LABELS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
df = load_data()
fdf = sidebar_filters(df)

chapter_header(55, "Bayesian Inference", part="XIII")

st.markdown(
    "Last chapter, Bayes' theorem helped us figure out which city a weather reading "
    "came from. That was about choosing between discrete hypotheses (Dallas? Houston? NYC?). "
    "This chapter tackles a different, arguably more important question."
)
st.markdown(
    "**The question**: What is the *true* average temperature of Dallas? Not the average "
    "of our particular sample of readings -- we can compute that in one line of code. "
    "The *true* long-run average. The number we would get if we measured the temperature "
    "every hour for a thousand years and averaged it all."
)
st.markdown(
    "We will never know this number exactly. But we can estimate it, and more importantly, "
    "we can quantify *how uncertain* we are about our estimate. That is what Bayesian "
    "inference gives us: not just a point estimate ('Dallas averages 20.3 C'), but a full "
    "probability distribution over all possible values of the true average ('there is a "
    "95% probability the true mean is between 19.8 and 20.8 C')."
)
st.markdown(
    "**Why this matters**: In weather science, the question 'what is the mean temperature?' "
    "is never as simple as it sounds. Is Dallas getting warmer over time? By how much? Are "
    "we confident? If the answer is 'the mean is 20.3 C plus or minus 5 C,' that is very "
    "different from 'the mean is 20.3 C plus or minus 0.1 C.' Bayesian inference gives you "
    "the uncertainty for free."
)

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "Bayesian Inference for Parameters",
    "Here is the key philosophical move: instead of treating the mean temperature of "
    "Dallas as a single fixed number that we are trying to pin down, we treat it as a "
    "<b>random variable</b> with its own probability distribution. This sounds like a "
    "weird ontological commitment ('the mean temperature is not uncertain -- it is "
    "whatever it is!'), but it turns out to be extraordinarily useful.<br><br>"
    "Before looking at data, you might believe Dallas averages around 25 C but you are "
    "not very sure -- maybe it could be anywhere from 15 to 35 C. That is your "
    "<b>prior distribution</b>: N(25, 10) -- a normal distribution centered at 25 C with "
    "standard deviation 10 C.<br><br>"
    "Then you observe 100 temperature readings from Dallas. The sample mean is 20.3 C. "
    "That data is highly informative. The <b>likelihood</b> says: 'Given a true mean of "
    "mu, how probable is a sample mean of 20.3 C?'<br><br>"
    "The <b>posterior distribution</b> combines prior and likelihood. With 100 observations, "
    "the data dominates: the posterior is now tightly centered near 20.3 C, with a standard "
    "deviation of maybe 0.5 C. Your initial guess of 25 C has been overwhelmed by the data.",
)

formula_box(
    "Normal-Normal Conjugate Update",
    r"\mu_{\text{post}} = \frac{\frac{\mu_0}{\sigma_0^2} + \frac{n \bar{x}}{\sigma^2}}"
    r"{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}, \quad "
    r"\sigma_{\text{post}}^2 = \frac{1}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}",
    "mu_0 and sigma_0 are the prior mean and std (your initial guess and uncertainty); "
    "x-bar is the sample mean from the data; sigma is the known data std; n is the "
    "sample size. The formula looks intimidating but it is just saying: 'average the "
    "prior and the data, weighted by how precise each is.' If your prior is vague "
    "(large sigma_0), the data gets almost all the weight. If you have very few data "
    "points (small n), the prior matters more.",
)

st.markdown("""
### Credible Interval vs Confidence Interval

This distinction sounds pedantic until you realize it changes everything about how you interpret results. Let me use a concrete example.

Say we compute a 95% interval for Dallas's mean temperature.

| | **Bayesian Credible Interval** | **Frequentist Confidence Interval** |
|---|---|---|
| **Interpretation** | "There is a 95% probability the true mean temperature of Dallas lies between 19.8 and 20.8 C" | "If we repeated this experiment many times, 95% of the resulting intervals would contain the true mean" |
| **Statement about** | The parameter (given the data you actually have) | The procedure (over hypothetical experiments you did not run) |
| **Requires** | A prior distribution | Only the data and sampling distribution |

The Bayesian credible interval says what everyone *thinks* a confidence interval says. Which is why so many textbooks spend pages explaining why confidence intervals do not mean what you think they mean, while credible intervals just... do. When a meteorologist says "we are 95% confident the temperature will be between 28 and 32 degrees," they are speaking Bayesian, whether they know it or not.
""")

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Bayesian Estimation of Mean Temperature
# ---------------------------------------------------------------------------
st.subheader("Interactive: Estimate Mean Temperature with Bayesian Updating")

st.markdown(
    "Here is the experiment. Pick a city, set your prior belief about its mean temperature, "
    "choose how many data points to observe, and watch the Bayesian update happen."
)
st.markdown(
    "**What the controls do:** 'Prior mean' is your starting guess for the city's average "
    "temperature. 'Prior std' is how uncertain you are (10 = very unsure, 1 = fairly "
    "confident). 'Number of data points' is how many hourly readings the model observes. "
    "'Credible interval level' sets the width of the posterior interval."
)
st.markdown(
    "**What to look for:** The orange dashed curve is your prior (starting belief). The "
    "red solid curve is the posterior (updated belief). The green dotted curve is the "
    "likelihood (what the data says). Watch how the posterior starts near the prior with "
    "few data points but migrates toward the data mean as you increase n. By n = 500, "
    "the prior barely matters."
)

col_ctrl, col_viz = st.columns([1, 2])

with col_ctrl:
    bayes_city = st.selectbox("City", CITY_LIST, key="bi_city")
    st.markdown("---")
    st.markdown("**Prior Beliefs** (your guess before seeing data)")
    prior_mean = st.slider(
        "Prior mean (deg C)", -10.0, 45.0, 25.0, 0.5, key="bi_prior_mean"
    )
    prior_std = st.slider(
        "Prior std (uncertainty)", 1.0, 30.0, 10.0, 0.5, key="bi_prior_std"
    )
    st.markdown("---")
    sample_size = st.slider(
        "Number of data points to observe", 5, 2000, 100, 5, key="bi_n"
    )
    cred_level = st.slider(
        "Credible interval level (%)", 80, 99, 95, key="bi_cred"
    )

city_temp = fdf[fdf["city"] == bayes_city]["temperature_c"].dropna().values
if len(city_temp) < sample_size:
    st.warning("Not enough data for this city/filter combination.")
    st.stop()

# True population stats
pop_mean = city_temp.mean()
pop_std = city_temp.std()

# Draw sample
rng = np.random.RandomState(42)
sample = rng.choice(city_temp, size=sample_size, replace=False)
sample_mean = sample.mean()
data_std = pop_std  # treat population std as known for conjugate update

# Posterior computation (Normal-Normal conjugate)
prior_prec = 1.0 / prior_std**2
data_prec = sample_size / data_std**2
post_prec = prior_prec + data_prec
post_var = 1.0 / post_prec
post_std = np.sqrt(post_var)
post_mean = (prior_mean * prior_prec + sample_mean * data_prec) / post_prec

# Credible interval
alpha = 1 - cred_level / 100
z = stats.norm.ppf(1 - alpha / 2)
cred_lower = post_mean - z * post_std
cred_upper = post_mean + z * post_std

# Frequentist CI for comparison
freq_se = sample.std(ddof=1) / np.sqrt(sample_size)
freq_lower = sample_mean - z * freq_se
freq_upper = sample_mean + z * freq_se

with col_viz:
    x_range = np.linspace(
        min(prior_mean - 3 * prior_std, post_mean - 4 * post_std, sample_mean - 4 * freq_se),
        max(prior_mean + 3 * prior_std, post_mean + 4 * post_std, sample_mean + 4 * freq_se),
        500,
    )

    fig = go.Figure()

    # Prior
    fig.add_trace(go.Scatter(
        x=x_range, y=stats.norm.pdf(x_range, prior_mean, prior_std),
        mode="lines", name="Prior",
        line=dict(color="#F4A261", width=2, dash="dash"),
        fill="tozeroy", fillcolor="rgba(244,162,97,0.15)",
    ))

    # Likelihood (scaled for visibility)
    lik_std = data_std / np.sqrt(sample_size)
    lik_y = stats.norm.pdf(x_range, sample_mean, lik_std)
    fig.add_trace(go.Scatter(
        x=x_range, y=lik_y,
        mode="lines", name="Likelihood (scaled)",
        line=dict(color="#2A9D8F", width=2, dash="dot"),
    ))

    # Posterior
    fig.add_trace(go.Scatter(
        x=x_range, y=stats.norm.pdf(x_range, post_mean, post_std),
        mode="lines", name="Posterior",
        line=dict(color="#E63946", width=3),
        fill="tozeroy", fillcolor="rgba(230,57,70,0.15)",
    ))

    # True mean
    fig.add_vline(x=pop_mean, line_dash="longdash", line_color="#264653", line_width=2,
                  annotation_text=f"True Mean: {pop_mean:.2f}")

    # Credible interval
    fig.add_vrect(x0=cred_lower, x1=cred_upper, fillcolor="rgba(230,57,70,0.08)",
                  line_width=0, annotation_text=f"{cred_level}% Credible Interval")

    fig.update_layout(
        xaxis_title="Temperature (deg C)", yaxis_title="Density",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    apply_common_layout(fig, title=f"Bayesian Update for Mean Temperature in {bayes_city}", height=500)
    st.plotly_chart(fig, use_container_width=True)

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Prior Mean", f"{prior_mean:.1f} C")
m2.metric("Sample Mean", f"{sample_mean:.2f} C")
m3.metric("Posterior Mean", f"{post_mean:.2f} C")
m4.metric("True Mean", f"{pop_mean:.2f} C")

st.markdown(
    f"**Interpreting the numbers:** Your prior guess was {prior_mean:.1f} C. The sample "
    f"of {sample_size} readings has a mean of {sample_mean:.2f} C. The posterior mean is "
    f"{post_mean:.2f} C -- a weighted average of these two, where the data contributes "
    f"{data_prec / post_prec:.1%} of the weight and the prior contributes "
    f"{prior_prec / post_prec:.1%}. The true population mean is {pop_mean:.2f} C. "
    f"Notice how the posterior is much closer to the data than to the prior -- with "
    f"n = {sample_size}, the data dominates."
)

st.markdown("#### Interval Comparison")
comp_df = pd.DataFrame({
    "Method": ["Bayesian Credible Interval", "Frequentist Confidence Interval"],
    "Lower": [f"{cred_lower:.2f}", f"{freq_lower:.2f}"],
    "Upper": [f"{cred_upper:.2f}", f"{freq_upper:.2f}"],
    "Width": [f"{cred_upper - cred_lower:.2f}", f"{freq_upper - freq_lower:.2f}"],
    "Centre": [f"{post_mean:.2f}", f"{sample_mean:.2f}"],
    "Contains True Mean": [
        cred_lower <= pop_mean <= cred_upper,
        freq_lower <= pop_mean <= freq_upper,
    ],
})
st.dataframe(comp_df, use_container_width=True, hide_index=True)

insight_box(
    f"The posterior mean ({post_mean:.2f} C) is a weighted average of the prior mean "
    f"({prior_mean:.1f} C) and the sample mean ({sample_mean:.2f} C). With n = {sample_size}, "
    f"the data contributes {data_prec / post_prec:.1%} of the weight -- the prior barely "
    f"matters. Try setting n = 5 and watch the prior's influence grow dramatically. "
    f"At n = 5, your starting guess actually matters. At n = 2000, it is essentially irrelevant. "
    f"This is the data-prior tug of war, and the data always wins eventually."
)

st.divider()

# ---------------------------------------------------------------------------
# 3. Prior Sensitivity: How Wrong Prior Gets Overwhelmed
# ---------------------------------------------------------------------------
st.subheader("Prior Sensitivity: Watching Data Overwhelm the Prior")

st.markdown(
    "This is the demo that makes Bayesian inference click. We start with a deliberately "
    "wrong prior -- a belief about the mean temperature that is nowhere near the truth -- "
    "and progressively add more data. Watch the posterior migrate from the prior's wrong "
    "answer to the data's right answer."
)
st.markdown(
    f"**What the slider does:** Set the wrong prior mean far from the true mean "
    f"({pop_mean:.1f} C for {bayes_city}). For maximum drama, try 0 C or 40 C. "
    f"Then watch the posterior curves at n = 1, 5, 10, 25, 50, 100, 250, 500, 1000 "
    f"all converge to the true mean despite starting from the wrong place."
)

wrong_prior_mean = st.slider(
    "Wrong prior mean (deg C)", -10.0, 45.0, 0.0, 1.0,
    help="Set this far from the true mean to see the effect", key="bi_wrong_prior",
)
wrong_prior_std = 5.0

sample_sizes = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
sample_sizes = [n for n in sample_sizes if n <= len(city_temp)]

# Use full data, draw progressively larger samples
all_samples = rng.choice(city_temp, size=max(sample_sizes), replace=False)

fig_sens = go.Figure()

# Prior
x_range_sens = np.linspace(
    min(wrong_prior_mean - 3 * wrong_prior_std, pop_mean - 15),
    max(wrong_prior_mean + 3 * wrong_prior_std, pop_mean + 15),
    500,
)

fig_sens.add_trace(go.Scatter(
    x=x_range_sens, y=stats.norm.pdf(x_range_sens, wrong_prior_mean, wrong_prior_std),
    mode="lines", name="Prior",
    line=dict(color="#F4A261", width=3, dash="dash"),
))

colorscale = ["#FFE0B2", "#FFCC80", "#FFB74D", "#FFA726", "#E63946", "#C62828",
              "#B71C1C", "#880E4F", "#4A148C"]

for i, n in enumerate(sample_sizes):
    s = all_samples[:n]
    s_mean = s.mean()
    pr_prec = 1.0 / wrong_prior_std**2
    d_prec = n / data_std**2
    p_prec = pr_prec + d_prec
    p_mean = (wrong_prior_mean * pr_prec + s_mean * d_prec) / p_prec
    p_std = np.sqrt(1.0 / p_prec)

    color = colorscale[i % len(colorscale)]
    fig_sens.add_trace(go.Scatter(
        x=x_range_sens, y=stats.norm.pdf(x_range_sens, p_mean, p_std),
        mode="lines", name=f"n={n}",
        line=dict(color=color, width=1.5),
    ))

fig_sens.add_vline(x=pop_mean, line_dash="longdash", line_color="#264653", line_width=2,
                   annotation_text=f"True: {pop_mean:.1f}")

fig_sens.update_layout(
    xaxis_title="Temperature (deg C)", yaxis_title="Density",
    legend_title="Posterior (n=...)",
)
apply_common_layout(fig_sens, title="Prior Being Overwhelmed by Data", height=500)
st.plotly_chart(fig_sens, use_container_width=True)

st.markdown(
    f"**Reading this chart:** The orange dashed curve is the wrong prior (centered at "
    f"{wrong_prior_mean:.0f} C). The colored curves show the posterior at increasing "
    f"sample sizes. At n = 1, the posterior is close to the prior -- one data point "
    f"cannot overcome a strongly held belief. By n = 25, the posterior has moved most of "
    f"the way toward the true mean ({pop_mean:.1f} C). By n = 100+, the posterior is "
    f"a tight spike centered on the truth, and the prior is a distant memory."
)

insight_box(
    "Even starting with a prior mean of {:.0f} C (far from the true mean of {:.1f} C), "
    "the posterior converges to the truth as sample size grows. By n = 100, the prior has "
    "almost no influence. This is the Bayesian promise: you do not need to be right at the "
    "start. You just need to be open to evidence. The data will eventually overwhelm any "
    "reasonable prior, and the speed depends on how wrong you were and how variable the "
    "data is.".format(wrong_prior_mean, pop_mean)
)

st.divider()

# ---------------------------------------------------------------------------
# 4. Sequential Updating Visualisation
# ---------------------------------------------------------------------------
st.subheader("Sequential Updating: Posterior Evolution")

st.markdown(
    "A different angle on the same idea. Instead of showing the full posterior curves, "
    "we track just two numbers as observations arrive one at a time: the posterior mean "
    "(our best guess) and the posterior standard deviation (our uncertainty)."
)
st.markdown(
    "**What to look for:** The left panel should show the posterior mean converging "
    f"toward the true value ({pop_mean:.1f} C). The right panel should show the posterior "
    "std monotonically decreasing -- each new observation reduces your uncertainty, "
    "even if only by a little. This is the rational learning process in action: more "
    "data = less uncertainty, always."
)

n_sequential = st.slider("Number of sequential observations", 5, 50, 20, key="bi_seq_n")
seq_obs = rng.choice(city_temp, size=min(n_sequential, len(city_temp)), replace=False)

frames_data = []
curr_mean = prior_mean
curr_std = prior_std

for i in range(len(seq_obs)):
    obs = seq_obs[i]
    # Update
    pr_prec = 1.0 / curr_std**2
    d_prec = 1.0 / data_std**2  # one observation at a time
    p_prec = pr_prec + d_prec
    new_mean = (curr_mean * pr_prec + obs * d_prec) / p_prec
    new_std = np.sqrt(1.0 / p_prec)

    frames_data.append({
        "step": i + 1,
        "observation": obs,
        "posterior_mean": new_mean,
        "posterior_std": new_std,
    })
    curr_mean = new_mean
    curr_std = new_std

frames_df = pd.DataFrame(frames_data)

fig_evo = make_subplots(rows=1, cols=2, subplot_titles=["Posterior Mean", "Posterior Std"])

fig_evo.add_trace(
    go.Scatter(
        x=frames_df["step"], y=frames_df["posterior_mean"],
        mode="lines+markers", name="Posterior Mean",
        line=dict(color="#E63946", width=2),
    ),
    row=1, col=1,
)
fig_evo.add_hline(y=pop_mean, line_dash="dash", line_color="#264653", row=1, col=1,
                  annotation_text=f"True: {pop_mean:.1f}")

fig_evo.add_trace(
    go.Scatter(
        x=frames_df["step"], y=frames_df["posterior_std"],
        mode="lines+markers", name="Posterior Std",
        line=dict(color="#2A9D8F", width=2),
    ),
    row=1, col=2,
)

fig_evo.update_xaxes(title_text="Observation #", row=1, col=1)
fig_evo.update_xaxes(title_text="Observation #", row=1, col=2)
fig_evo.update_yaxes(title_text="Mean (deg C)", row=1, col=1)
fig_evo.update_yaxes(title_text="Std (deg C)", row=1, col=2)
fig_evo.update_layout(template="plotly_white", height=400, margin=dict(t=60, b=40), showlegend=False)
st.plotly_chart(fig_evo, use_container_width=True)

st.markdown(
    f"After {len(seq_obs)} observations: posterior mean = **{curr_mean:.2f} C**, "
    f"posterior std = **{curr_std:.3f} C** (true mean = {pop_mean:.2f} C). "
    "The posterior mean has converged near the truth, and the uncertainty (std) has shrunk "
    f"from {prior_std:.1f} C to {curr_std:.3f} C -- a {prior_std / curr_std:.0f}x reduction. "
    "Each new observation tightens the estimate. This is what rational learning looks like."
)

st.divider()

# ---------------------------------------------------------------------------
# 5. Compare Across Cities
# ---------------------------------------------------------------------------
st.subheader("Bayesian Estimates Across All Cities")

st.markdown(
    f"Using the same prior for every city (N({prior_mean:.0f}, {prior_std:.0f})) -- "
    "which is deliberately a bit wrong for most of them -- we run the Bayesian update "
    "with 500 observations per city. The 'Data Weight' column tells you how much the "
    "prior matters: spoiler, with 500 observations, it barely matters at all."
)

comparison_rows = []
for city in CITY_LIST:
    c_data = fdf[fdf["city"] == city]["temperature_c"].dropna().values
    if len(c_data) < 10:
        continue
    c_mean = c_data.mean()
    c_std = c_data.std()
    n_c = min(500, len(c_data))
    c_sample = rng.choice(c_data, size=n_c, replace=False)

    pr_prec = 1.0 / prior_std**2
    d_prec = n_c / c_std**2
    p_prec = pr_prec + d_prec
    p_mean = (prior_mean * pr_prec + c_sample.mean() * d_prec) / p_prec
    p_std = np.sqrt(1.0 / p_prec)
    z95 = 1.96

    comparison_rows.append({
        "City": city,
        "True Mean": f"{c_mean:.2f}",
        "Prior Mean": f"{prior_mean:.1f}",
        "Posterior Mean": f"{p_mean:.2f}",
        "Posterior Std": f"{p_std:.3f}",
        f"95% Credible Interval": f"({p_mean - z95 * p_std:.2f}, {p_mean + z95 * p_std:.2f})",
        "Data Weight": f"{d_prec / p_prec:.1%}",
    })

if comparison_rows:
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

st.markdown(
    "**Reading the table:** Look at the 'Data Weight' column -- it should be 99%+ for "
    "every city. That means the prior (your initial guess of "
    f"{prior_mean:.0f} C) contributes less than 1% to the posterior. The 95% credible "
    "interval is narrow (typically 1-2 C wide), meaning we are quite precise about each "
    "city's mean. NYC's true mean (~13 C) and LA's true mean (~18 C) are easily "
    "distinguished, even though we started with the same prior for both."
)

st.divider()

# ---------------------------------------------------------------------------
# 6. Code Example
# ---------------------------------------------------------------------------
code_example("""
import numpy as np
from scipy import stats

# Observed data
data = city_df['temperature_c'].values
n = len(data)
x_bar = data.mean()
sigma = data.std()  # treat as known

# Prior
prior_mean = 25.0
prior_std = 10.0

# Conjugate Normal-Normal update
prior_prec = 1 / prior_std**2
data_prec = n / sigma**2
post_prec = prior_prec + data_prec

post_mean = (prior_mean * prior_prec + x_bar * data_prec) / post_prec
post_std = np.sqrt(1 / post_prec)

# 95% Credible Interval
cred_lower = post_mean - 1.96 * post_std
cred_upper = post_mean + 1.96 * post_std

print(f"Posterior: N({post_mean:.2f}, {post_std:.4f})")
print(f"95% CI: ({cred_lower:.2f}, {cred_upper:.2f})")
print(f"Data weight: {data_prec/post_prec:.1%}")
""")

st.divider()

# ---------------------------------------------------------------------------
# 7. Quiz
# ---------------------------------------------------------------------------
quiz(
    "In Bayesian inference, a credible interval:",
    [
        "Has the same interpretation as a frequentist confidence interval",
        "Gives the probability that the parameter lies within the interval, given the data",
        "Requires no prior distribution",
        "Is always wider than a confidence interval",
    ],
    correct_idx=1,
    explanation=(
        "This is the big payoff of the Bayesian approach. A 95% credible interval for "
        "Dallas's mean temperature says: 'Given the data we observed, there is a 95% "
        "probability the true mean is between 19.8 and 20.8 C.' That is the natural "
        "interpretation everyone wants. The frequentist confidence interval says something "
        "subtly different: 'If we repeated this experiment many times, 95% of our intervals "
        "would contain the true mean.' Note that the frequentist statement is about the "
        "procedure, not about this specific interval. The Bayesian statement is about this "
        "specific interval, which is usually what you want to know. In practice, with "
        "large samples and weak priors, the two intervals are nearly identical -- the "
        "philosophical difference matters most with small samples or strong priors."
    ),
    key="ch55_quiz1",
)

quiz(
    "What happens to the posterior as sample size increases?",
    [
        "It becomes more influenced by the prior",
        "It stays the same regardless of data",
        "It concentrates around the true parameter value (data dominates)",
        "It becomes wider",
    ],
    correct_idx=2,
    explanation=(
        "As n grows, data precision (n / sigma^2) increases while prior precision (1 / sigma_0^2) "
        "stays fixed. The data's weight in the posterior grows from almost nothing (at n = 1) to "
        "almost everything (at n = 500+). In our demo, with n = 100 and a prior std of 10 C, "
        "the data already contributes 99%+ of the weight. The posterior concentrates into a "
        "narrow spike centered at the sample mean, and the prior is irrelevant. This is "
        "why Bayesian inference is self-correcting: no matter what prior you start with, "
        "enough data brings you to the truth."
    ),
    key="ch55_quiz2",
)

quiz(
    "You are estimating Dallas's mean temperature. Your prior is N(25, 10) and you "
    "observe n = 5 readings with a sample mean of 20 C. Roughly where will the "
    "posterior mean be?",
    [
        "Exactly at 25 C (the prior dominates with only 5 observations)",
        "Exactly at 20 C (the data always dominates)",
        "Between 20 and 25, closer to 20 but noticeably pulled toward 25",
        "Outside the range 20-25",
    ],
    correct_idx=2,
    explanation=(
        "With only 5 observations, the prior still has meaningful influence. The posterior "
        "mean is a weighted average of the prior mean (25 C) and the sample mean (20 C). "
        "With Dallas's temperature std around 10 C, the data precision at n = 5 is "
        "5 / (10^2) = 0.05, while the prior precision is 1 / (10^2) = 0.01. Data weight = "
        "0.05 / (0.05 + 0.01) = 83%. So the posterior mean is roughly 0.83 * 20 + 0.17 * 25 "
        "= 20.85 C. The data mostly wins even at n = 5, but the prior pulls it about 0.85 C "
        "toward 25. At n = 50, that pull would shrink to about 0.09 C -- essentially nothing. "
        "Try it in the interactive demo above to verify."
    ),
    key="ch55_quiz3",
)

st.divider()

# ---------------------------------------------------------------------------
# 8. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Bayesian inference treats parameters (like Dallas's mean temperature) as random "
    "variables with probability distributions. Your prior might be N(25, 10) -- a guess "
    "of 25 C with high uncertainty. After 100 observations, the posterior might be "
    "N(20.3, 0.5) -- precise and data-driven.",
    "The posterior is a **weighted average** of prior and data. With n = 5, the prior "
    "contributes ~17% of the weight. With n = 100, the prior contributes ~1%. With "
    "n = 1000, the prior is irrelevant. The data always wins eventually.",
    "**Credible intervals** have a direct probability interpretation: '95% probability "
    "the true mean is in this range.' This is what everyone thinks confidence intervals "
    "mean, but only credible intervals actually deliver.",
    "Wrong priors are **self-correcting**. Starting at 0 C or 40 C for Dallas does not "
    "matter -- by n = 100, the posterior converges to the true mean (~20 C) regardless. "
    "You need to be open to evidence, not correct from the start.",
    "The Normal-Normal conjugate model gives **closed-form updates**, making the math "
    "transparent. Posterior mean = weighted average of prior mean and sample mean, "
    "with weights proportional to their respective precisions.",
])

navigation(
    prev_label="Ch 54: Bayes' Theorem",
    prev_page="54_Bayes_Theorem.py",
    next_label="Ch 56: Probabilistic Programming",
    next_page="56_Probabilistic_Programming.py",
)
