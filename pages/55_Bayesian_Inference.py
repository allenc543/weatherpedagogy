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
st.set_page_config(page_title="Ch 55: Bayesian Inference", layout="wide")
df = load_data()
fdf = sidebar_filters(df)

chapter_header(55, "Bayesian Inference", part="XIII")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "Bayesian Inference for Parameters",
    "Here is the key philosophical move in Bayesian inference: instead of treating "
    "parameters (like the mean temperature of a city) as fixed but unknown numbers that "
    "we are trying to pin down, we treat them as <b>random variables</b> with their own "
    "probability distributions. This might sound like a weird ontological commitment, but "
    "it turns out to be extraordinarily useful.<br><br>"
    "<b>Prior distribution</b>: your beliefs about the parameter before looking at data. "
    "Maybe you think Dallas averages 25 C but you are not very sure.<br>"
    "<b>Likelihood</b>: how probable is the observed data given a specific parameter value?<br>"
    "<b>Posterior distribution</b>: your updated beliefs after seeing data. A full "
    "distribution, not just a point estimate -- which means you get uncertainty for free.<br><br>"
    "For a normal likelihood with known variance and a normal prior on the mean, the "
    "posterior is also normal (this is called a conjugate prior, and it is one of the few "
    "cases where the math works out neatly). The posterior mean is a <em>weighted average</em> "
    "of the prior mean and the sample mean, where the weights depend on how confident you are "
    "in each.",
)

formula_box(
    "Normal-Normal Conjugate Update",
    r"\mu_{\text{post}} = \frac{\frac{\mu_0}{\sigma_0^2} + \frac{n \bar{x}}{\sigma^2}}"
    r"{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}, \quad "
    r"\sigma_{\text{post}}^2 = \frac{1}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}",
    "mu_0 and sigma_0 are the prior mean and std; x-bar is the sample mean; "
    "sigma is the known data std; n is the sample size. The formula looks intimidating "
    "but it is just saying: 'average the prior and the data, weighted by how precise each is.'",
)

st.markdown("""
### Credible Interval vs Confidence Interval

This is one of those distinctions that sounds pedantic until you realize it changes everything
about how you interpret results.

| | **Bayesian Credible Interval** | **Frequentist Confidence Interval** |
|---|---|---|
| **Interpretation** | There is a 95% probability the parameter lies in this interval | 95% of such intervals (constructed from repeated samples) would contain the true parameter |
| **Statement about** | The parameter (given the data you actually have) | The procedure (over hypothetical experiments you didn't run) |
| **Requires** | A prior distribution | Only the data and sampling distribution |

The Bayesian credible interval says what everyone *thinks* a confidence interval says. Which
is why so many textbooks spend pages explaining why confidence intervals do not mean what
you think they mean, while credible intervals just... do.
""")

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Bayesian Estimation of Mean Temperature
# ---------------------------------------------------------------------------
st.subheader("Interactive: Estimate Mean Temperature with Bayesian Updating")

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
    "The posterior mean is a weighted average of the prior mean and the sample mean. "
    f"With n={sample_size}, the data contributes {data_prec / post_prec:.1%} of the weight "
    f"and the prior contributes {prior_prec / post_prec:.1%}. "
    "Try cranking the sample size up and watch the prior's influence shrink to nothing."
)

st.divider()

# ---------------------------------------------------------------------------
# 3. Prior Sensitivity: How Wrong Prior Gets Overwhelmed
# ---------------------------------------------------------------------------
st.subheader("Prior Sensitivity: Watching Data Overwhelm the Prior")

st.markdown(
    "This is the demo I find most satisfying. We start with an intentionally wrong prior -- "
    "a belief about the mean temperature that is nowhere near the truth -- and progressively "
    "add more data. The posterior starts near the prior, then migrates inexorably toward reality. "
    "By n=100 or so, the prior has been effectively silenced."
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

insight_box(
    "Even starting with a prior mean of {:.0f} deg C (far from the true mean of {:.1f} deg C), "
    "the posterior converges to the truth as sample size grows. "
    "By n=100, the prior has almost no influence. This is the Bayesian promise: "
    "you do not need to be right, you just need to be open to evidence.".format(wrong_prior_mean, pop_mean)
)

st.divider()

# ---------------------------------------------------------------------------
# 4. Sequential Updating Visualisation
# ---------------------------------------------------------------------------
st.subheader("Sequential Updating: Posterior Evolution")

st.markdown(
    "Here is the same idea from a different angle. Instead of showing the full posterior "
    "curves, we track how the posterior mean and standard deviation evolve as we process "
    "observations one at a time. Watch the mean converge and the uncertainty shrink."
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
    "The uncertainty keeps shrinking with each new observation, which is exactly what you "
    "would want from a rational learning process."
)

st.divider()

# ---------------------------------------------------------------------------
# 5. Compare Across Cities
# ---------------------------------------------------------------------------
st.subheader("Bayesian Estimates Across All Cities")

st.markdown(
    "Using the same prior for every city -- which is deliberately a bit wrong for most of them -- "
    "see how the posterior adapts to each city's data. The data weight column tells you "
    "how much the prior matters: spoiler, it barely matters at all."
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
    explanation="This is the big payoff of the Bayesian approach. A credible interval directly "
                "answers the question everyone actually wants answered: 'Given what I observed, "
                "where is the parameter?' The frequentist CI answers a subtly different, less "
                "intuitive question about hypothetical repeated experiments.",
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
    explanation="As n grows, data precision increases and the posterior concentrates "
                "around the maximum likelihood estimate, overwhelming the prior. "
                "The prior becomes a rounding error.",
    key="ch55_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 8. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Bayesian inference treats parameters as random variables with distributions -- a philosophical choice that buys you honest uncertainty quantification.",
    "The posterior is a compromise between the prior and the likelihood (data). With enough data, the compromise is not very compromising -- the data wins.",
    "With more data, the posterior concentrates around the true value regardless of the prior. You can start wrong and end up right.",
    "Credible intervals have a direct probability interpretation, unlike frequentist CIs. They answer the question people actually want to ask.",
    "The Normal-Normal conjugate model provides closed-form posterior updates, which makes the math tractable and the intuition transparent.",
])

navigation(
    prev_label="Ch 54: Bayes' Theorem",
    prev_page="54_Bayes_Theorem.py",
    next_label="Ch 56: Probabilistic Programming",
    next_page="56_Probabilistic_Programming.py",
)
