"""Chapter 56: Probabilistic Programming — PyMC, MCMC, hierarchical models."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, scatter_chart
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)
from utils.constants import CITY_LIST, CITY_COLORS, FEATURE_LABELS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Ch 56: Probabilistic Programming", layout="wide")
df = load_data()
fdf = sidebar_filters(df)

chapter_header(56, "Probabilistic Programming", part="XIII")

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "What Is Probabilistic Programming?",
    "Remember how in the last chapter we got nice closed-form posterior updates because "
    "we used conjugate priors? That was lovely, but it only works for a tiny fraction of "
    "real-world models. The moment your model gets even slightly complicated -- say, a "
    "hierarchical structure or a non-linear link function -- the math stops being tractable.<br><br>"
    "Probabilistic programming languages (PPLs) like <b>PyMC</b>, Stan, and Pyro solve this by "
    "letting you specify your model in code, then automatically performing <b>Bayesian inference</b> "
    "using algorithms like MCMC (Markov Chain Monte Carlo). You describe the generative story of "
    "your data, and the PPL figures out the posterior. It is, frankly, a little magical.<br><br>"
    "The workflow is always the same:<br>"
    "1. Define priors for each parameter<br>"
    "2. Define the likelihood (how data relates to parameters)<br>"
    "3. Let the PPL sample from the posterior automatically",
)

concept_box(
    "MCMC: Markov Chain Monte Carlo",
    "Here is the core idea: if you cannot compute the posterior analytically, you can "
    "instead generate <em>samples</em> from it by constructing a cleverly designed random walk. "
    "Algorithms like the <b>No-U-Turn Sampler (NUTS)</b> are particularly good at this.<br><br>"
    "The Markov chain's stationary distribution is the posterior, so if you run it long enough "
    "and check that it has converged, your samples are as good as having the closed-form answer. "
    "Key diagnostics to watch for:<br>"
    "- <b>Trace plots</b>: should look like 'hairy caterpillars' (well-mixed). If they look like "
    "snakes trying to escape a box, something has gone wrong.<br>"
    "- <b>R-hat</b>: should be close to 1.0. If it is not, your chains have not converged and "
    "your results are not to be trusted.<br>"
    "- <b>Effective sample size</b>: should be large, indicating low autocorrelation",
)

st.markdown("""
### The PyMC Workflow

```python
import pymc as pm

with pm.Model() as model:
    # 1. Priors
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    slope = pm.Normal('slope', mu=0, sigma=1)
    sigma = pm.HalfNormal('sigma', sigma=5)

    # 2. Likelihood
    mu = intercept + slope * x_data
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_data)

    # 3. Inference
    trace = pm.sample(1000, cores=2)
```

That is it. Three blocks of code and you have posterior distributions for all your parameters.
The PPL handles the MCMC sampling, adaptation, convergence checking, and diagnostics.
""")

warning_box(
    "PyMC requires a working installation with a compatible backend (JAX or C compiler). "
    "In this chapter, we simulate what PyMC would produce using analytical solutions and "
    "Metropolis-Hastings sampling to demonstrate the concepts without requiring PyMC installation. "
    "Think of it as the pedagogical equivalent of a flight simulator."
)

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Bayesian Linear Model (temp ~ humidity)
# ---------------------------------------------------------------------------
st.subheader("Interactive: Bayesian Linear Regression (Temperature ~ Humidity)")

col_ctrl, col_data = st.columns([1, 2])

with col_ctrl:
    pp_city = st.selectbox("City", CITY_LIST, key="pp_city")
    n_sample = st.slider("Data sample size", 100, 2000, 500, 50, key="pp_n")
    n_mcmc = st.slider("MCMC samples", 500, 5000, 2000, 500, key="pp_mcmc")
    prior_slope_std = st.slider("Prior std for slope", 0.1, 5.0, 1.0, 0.1, key="pp_slope_prior")

city_data = fdf[fdf["city"] == pp_city][["temperature_c", "relative_humidity_pct"]].dropna()
if len(city_data) < 50:
    st.warning("Not enough data for this city. Adjust filters.")
    st.stop()

rng = np.random.RandomState(42)
sample_idx = rng.choice(len(city_data), size=min(n_sample, len(city_data)), replace=False)
sample_data = city_data.iloc[sample_idx].copy()
x_data = sample_data["relative_humidity_pct"].values
y_data = sample_data["temperature_c"].values

# Center data for better MCMC
x_mean, x_std_val = x_data.mean(), x_data.std()
y_mean = y_data.mean()
x_centered = (x_data - x_mean) / x_std_val

with col_data:
    fig_scatter = scatter_chart(
        sample_data, x="relative_humidity_pct", y="temperature_c",
        color=None, title=f"Temperature vs Humidity in {pp_city}",
        opacity=0.3,
    )
    # Override the color since we have a single city
    fig_scatter.data[0].marker.color = CITY_COLORS.get(pp_city, "#2A9D8F")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------------------------
# 3. Metropolis-Hastings MCMC Sampler
# ---------------------------------------------------------------------------
st.subheader("MCMC Sampling (Metropolis-Hastings)")

st.markdown(
    "We implement a simple Metropolis-Hastings sampler to estimate the intercept, "
    "slope, and noise of a linear model: **temp = intercept + slope * humidity + noise**. "
    "This is the same thing PyMC does under the hood, just with a less sophisticated algorithm "
    "and more educational transparency."
)

@st.cache_data(show_spinner="Running MCMC sampler...")
def run_metropolis(x, y, n_samples, prior_slope_std, seed=42):
    """Simple Metropolis-Hastings for linear regression."""
    rng_mh = np.random.RandomState(seed)

    # Log posterior function
    def log_posterior(intercept, slope, log_sigma):
        sigma = np.exp(log_sigma)
        # Priors
        lp = stats.norm.logpdf(intercept, 0, 10)
        lp += stats.norm.logpdf(slope, 0, prior_slope_std)
        lp += stats.norm.logpdf(log_sigma, 0, 2)  # prior on log(sigma)
        # Likelihood
        mu = intercept + slope * x
        lp += np.sum(stats.norm.logpdf(y, mu, sigma))
        return lp

    # Initial values (OLS)
    from numpy.linalg import lstsq
    A = np.column_stack([np.ones_like(x), x])
    ols_params, _, _, _ = lstsq(A, y, rcond=None)
    residuals = y - A @ ols_params
    init_sigma = np.log(np.std(residuals) + 0.01)

    current = np.array([ols_params[0], ols_params[1], init_sigma])
    current_lp = log_posterior(*current)

    # Proposal scales
    proposal_std = np.array([0.1, 0.01, 0.05])

    samples = []
    accepted = 0

    for i in range(n_samples + 500):  # 500 burn-in
        proposal = current + rng_mh.randn(3) * proposal_std
        prop_lp = log_posterior(*proposal)

        if np.log(rng_mh.rand()) < prop_lp - current_lp:
            current = proposal
            current_lp = prop_lp
            if i >= 500:
                accepted += 1

        if i >= 500:
            samples.append(current.copy())

    samples = np.array(samples)
    acceptance_rate = accepted / n_samples
    return samples, acceptance_rate

samples, acc_rate = run_metropolis(x_centered, y_data, n_mcmc, prior_slope_std)

intercept_samples = samples[:, 0]
slope_samples = samples[:, 1]  # on centred scale
sigma_samples = np.exp(samples[:, 2])

# Convert slope back to original scale
slope_original = slope_samples / x_std_val
intercept_original = intercept_samples - slope_samples * x_mean / x_std_val

st.markdown(f"**Acceptance rate:** {acc_rate:.1%} (target: 20-50%)")

# ---------------------------------------------------------------------------
# 4. Trace Plots
# ---------------------------------------------------------------------------
st.subheader("Trace Plots and Posterior Distributions")

fig_trace = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Intercept Trace", "Intercept Posterior",
        "Slope Trace", "Slope Posterior",
        "Sigma Trace", "Sigma Posterior",
    ],
)

params = [
    ("Intercept", intercept_original),
    ("Slope", slope_original),
    ("Sigma", sigma_samples),
]

colors_trace = ["#E63946", "#2A9D8F", "#7209B7"]

for i, (name, vals) in enumerate(params):
    row = i + 1
    # Trace
    fig_trace.add_trace(
        go.Scatter(y=vals, mode="lines", line=dict(color=colors_trace[i], width=0.5),
                   showlegend=False),
        row=row, col=1,
    )
    fig_trace.add_hline(y=np.mean(vals), line_dash="dash", line_color="black",
                        row=row, col=1)

    # Posterior histogram
    fig_trace.add_trace(
        go.Histogram(x=vals, nbinsx=50, marker_color=colors_trace[i],
                     opacity=0.7, showlegend=False),
        row=row, col=2,
    )
    # Mean and 95% CI lines
    lo, hi = np.percentile(vals, [2.5, 97.5])
    fig_trace.add_vline(x=np.mean(vals), line_dash="solid", line_color="black",
                        row=row, col=2)
    fig_trace.add_vline(x=lo, line_dash="dash", line_color="gray", row=row, col=2)
    fig_trace.add_vline(x=hi, line_dash="dash", line_color="gray", row=row, col=2)

fig_trace.update_layout(template="plotly_white", height=700, margin=dict(t=40, b=40),
                        showlegend=False)
st.plotly_chart(fig_trace, use_container_width=True)

# Summary table
summary_rows = []
for name, vals in params:
    lo, hi = np.percentile(vals, [2.5, 97.5])
    summary_rows.append({
        "Parameter": name,
        "Mean": f"{np.mean(vals):.4f}",
        "Std": f"{np.std(vals):.4f}",
        "2.5%": f"{lo:.4f}",
        "97.5%": f"{hi:.4f}",
    })

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# OLS comparison
from numpy.linalg import lstsq
A_full = np.column_stack([np.ones_like(x_data), x_data])
ols_params, _, _, _ = lstsq(A_full, y_data, rcond=None)

st.markdown(
    f"**OLS estimates for comparison:** intercept = {ols_params[0]:.4f}, "
    f"slope = {ols_params[1]:.4f}"
)

insight_box(
    "The Bayesian posterior means are very close to the OLS estimates, which is exactly what "
    "you would expect -- we used relatively uninformative priors and have enough data. The key "
    "advantage of the Bayesian approach is not a different point estimate; it is the full "
    "posterior distribution and credible intervals. You do not just know the slope is probably "
    "around -0.1; you know the probability it is between -0.08 and -0.12."
)

st.divider()

# ---------------------------------------------------------------------------
# 5. Posterior Predictive Plot
# ---------------------------------------------------------------------------
st.subheader("Posterior Predictive: Regression Lines from the Posterior")

x_pred = np.linspace(x_data.min(), x_data.max(), 100)

fig_pred = go.Figure()

# Scatter of data
fig_pred.add_trace(go.Scatter(
    x=x_data, y=y_data, mode="markers",
    marker=dict(color=CITY_COLORS.get(pp_city, "#2A9D8F"), opacity=0.2, size=4),
    name="Data",
))

# Draw 100 posterior regression lines
n_lines = min(100, len(intercept_original))
idx = rng.choice(len(intercept_original), size=n_lines, replace=False)
for j in idx:
    y_pred = intercept_original[j] + slope_original[j] * x_pred
    fig_pred.add_trace(go.Scatter(
        x=x_pred, y=y_pred, mode="lines",
        line=dict(color="#E63946", width=0.3), opacity=0.15,
        showlegend=False,
    ))

# Mean line
mean_int = np.mean(intercept_original)
mean_slope = np.mean(slope_original)
fig_pred.add_trace(go.Scatter(
    x=x_pred, y=mean_int + mean_slope * x_pred,
    mode="lines", line=dict(color="#264653", width=3),
    name="Posterior Mean",
))

fig_pred.update_layout(
    xaxis_title="Humidity (%)", yaxis_title="Temperature (deg C)",
)
apply_common_layout(fig_pred, title="Posterior Predictive Regression Lines", height=500)
st.plotly_chart(fig_pred, use_container_width=True)

insight_box(
    "Each faint red line represents one plausible regression line drawn from the posterior. "
    "The spread of these lines is your uncertainty made visible. Where the fan is narrow, "
    "you are confident. Where it is wide, you are not. "
    "This is a key advantage of Bayesian regression: you get a distribution of predictions, "
    "not just a single best-fit line pretending to be certain."
)

st.divider()

# ---------------------------------------------------------------------------
# 6. Hierarchical Model Concept
# ---------------------------------------------------------------------------
st.subheader("Hierarchical Models: Partial Pooling Across Cities")

concept_box(
    "Hierarchical (Multilevel) Models",
    "When we have data from multiple cities, we face an interesting modelling dilemma with "
    "three options, two of which are clearly wrong:<br><br>"
    "<b>Complete pooling</b>: ignore city differences entirely, fit one model to all data. "
    "This throws away real information.<br>"
    "<b>No pooling</b>: fit a separate model for each city independently. This ignores the "
    "fact that cities share the same planet and similar physics.<br>"
    "<b>Partial pooling (hierarchical)</b>: each city has its own parameters, but they "
    "are drawn from a shared group-level distribution. Cities with less data get "
    "'shrunk' toward the group mean -- borrowing strength from other cities.<br><br>"
    "Partial pooling is the Goldilocks option, and it is one of the most practically useful "
    "ideas in all of Bayesian statistics.",
)

formula_box(
    "Hierarchical Model for City Temperatures",
    r"\mu_i \sim \mathcal{N}(\mu_{\text{global}}, \tau^2), \quad "
    r"y_{ij} \sim \mathcal{N}(\mu_i, \sigma^2)",
    "mu_i is the mean temperature for city i; mu_global and tau describe "
    "the group-level distribution; sigma is the within-city noise. The key insight "
    "is that each city's mu_i is regularized toward the global mean.",
)

# Demonstrate shrinkage
st.markdown("#### Shrinkage Effect: Partial Pooling vs No Pooling")

city_means = {}
city_ns = {}
for city in CITY_LIST:
    c_data = fdf[fdf["city"] == city]["temperature_c"].dropna()
    if len(c_data) > 0:
        city_means[city] = c_data.mean()
        city_ns[city] = len(c_data)

if len(city_means) >= 2:
    active = list(city_means.keys())
    grand_mean = np.mean(list(city_means.values()))
    city_stds = {c: fdf[fdf["city"] == c]["temperature_c"].std() for c in active}

    # Simulate partial pooling shrinkage
    # Use James-Stein-like estimator for demonstration
    tau_hat = np.std(list(city_means.values()))
    sigma_hat = np.mean(list(city_stds.values()))

    shrunk_means = {}
    for city in active:
        n_c = city_ns[city]
        shrinkage = (sigma_hat**2 / n_c) / (sigma_hat**2 / n_c + tau_hat**2)
        shrunk_means[city] = (1 - shrinkage) * city_means[city] + shrinkage * grand_mean

    fig_shrink = go.Figure()

    # No pooling (individual means)
    fig_shrink.add_trace(go.Scatter(
        x=[city_means[c] for c in active],
        y=active,
        mode="markers",
        marker=dict(size=12, color=[CITY_COLORS.get(c, "#888") for c in active],
                    symbol="circle"),
        name="No Pooling (city mean)",
    ))

    # Partial pooling
    fig_shrink.add_trace(go.Scatter(
        x=[shrunk_means[c] for c in active],
        y=active,
        mode="markers",
        marker=dict(size=12, color=[CITY_COLORS.get(c, "#888") for c in active],
                    symbol="diamond"),
        name="Partial Pooling (shrunk)",
    ))

    # Arrows showing shrinkage
    for city in active:
        fig_shrink.add_annotation(
            x=shrunk_means[city], y=city,
            ax=city_means[city], ay=city,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
            arrowcolor="#888",
        )

    # Grand mean
    fig_shrink.add_vline(x=grand_mean, line_dash="dash", line_color="#264653",
                         annotation_text=f"Grand Mean: {grand_mean:.1f}")

    fig_shrink.update_layout(
        xaxis_title="Mean Temperature (deg C)",
    )
    apply_common_layout(fig_shrink, title="Shrinkage: Partial Pooling Pulls Estimates Toward Grand Mean",
                        height=400)
    st.plotly_chart(fig_shrink, use_container_width=True)

    shrink_df = pd.DataFrame({
        "City": active,
        "No Pooling Mean": [f"{city_means[c]:.2f}" for c in active],
        "Partial Pooling Mean": [f"{shrunk_means[c]:.2f}" for c in active],
        "Shrinkage Amount": [f"{abs(city_means[c] - shrunk_means[c]):.2f}" for c in active],
        "N Observations": [city_ns[c] for c in active],
    })
    st.dataframe(shrink_df, use_container_width=True, hide_index=True)

    insight_box(
        "Partial pooling 'shrinks' each city's estimate toward the grand mean. "
        "Cities with fewer observations are shrunk more -- they borrow more strength "
        "from the group. This is not a bug; it is a feature. It is the model saying: "
        "'I do not have much data for this city, so I will hedge toward what the other cities suggest.' "
        "This reduces overfitting and typically improves out-of-sample predictions."
    )

st.divider()

# ---------------------------------------------------------------------------
# 7. Code Example
# ---------------------------------------------------------------------------
code_example("""
import pymc as pm
import arviz as az

# Load data for one city
x = city_df['relative_humidity_pct'].values
y = city_df['temperature_c'].values

# Build model
with pm.Model() as linear_model:
    # Priors
    intercept = pm.Normal('intercept', mu=20, sigma=10)
    slope = pm.Normal('slope', mu=-0.1, sigma=1)
    sigma = pm.HalfNormal('sigma', sigma=5)

    # Linear model
    mu = intercept + slope * x

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # Sample
    trace = pm.sample(2000, tune=1000, cores=2, random_seed=42)

# Diagnostics
az.summary(trace)
az.plot_trace(trace)
az.plot_posterior(trace)

# ------- Hierarchical Model -------
with pm.Model() as hierarchical_model:
    # Group-level priors
    mu_global = pm.Normal('mu_global', mu=20, sigma=10)
    tau = pm.HalfNormal('tau', sigma=5)
    sigma = pm.HalfNormal('sigma', sigma=5)

    # City-level parameters
    mu_city = pm.Normal('mu_city', mu=mu_global, sigma=tau,
                        shape=n_cities)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu_city[city_idx],
                      sigma=sigma, observed=y_all)

    trace_hier = pm.sample(2000, tune=1000, cores=2)
""")

st.divider()

# ---------------------------------------------------------------------------
# 8. Quiz
# ---------------------------------------------------------------------------
quiz(
    "What does MCMC stand for and what does it produce?",
    [
        "Maximum Chain Monte Carlo -- produces maximum likelihood estimates",
        "Markov Chain Monte Carlo -- produces samples from the posterior distribution",
        "Multiple Cluster Monte Carlo -- produces cluster assignments",
        "Markov Chain Monte Carlo -- produces point estimates only",
    ],
    correct_idx=1,
    explanation="MCMC generates samples from the posterior distribution by constructing "
                "a Markov chain whose stationary distribution is the target posterior. "
                "The samples let you approximate any property of the posterior you want -- "
                "means, medians, credible intervals, the works.",
    key="ch56_quiz1",
)

quiz(
    "In a hierarchical model, partial pooling means:",
    [
        "All cities share exactly the same parameters",
        "Each city has completely independent parameters",
        "City parameters are drawn from a shared group distribution, shrinking toward the group mean",
        "Only half the data is used for estimation",
    ],
    correct_idx=2,
    explanation="Partial pooling means city-level parameters are linked through a group-level "
                "distribution. Cities with less data are shrunk more toward the group mean. "
                "It is a principled way to balance individual estimation with information sharing.",
    key="ch56_quiz2",
)

st.divider()

# ---------------------------------------------------------------------------
# 9. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "Probabilistic programming languages automate Bayesian inference via MCMC, saving you from having to derive posteriors by hand (which is good, because most posteriors do not have closed forms).",
    "MCMC produces samples from the posterior -- check convergence with trace plots and R-hat before trusting anything.",
    "Bayesian regression gives posterior distributions over parameters, not just point estimates. The uncertainty is a feature, not a limitation.",
    "Posterior predictive checks show the range of plausible model predictions, making uncertainty visible and actionable.",
    "Hierarchical models enable partial pooling, borrowing strength across groups -- one of the most practically useful ideas in modern statistics.",
])

navigation(
    prev_label="Ch 55: Bayesian Inference",
    prev_page="55_Bayesian_Inference.py",
    next_label="Ch 57: Statistical Anomaly Detection",
    next_page="57_Statistical_Anomaly_Detection.py",
)
