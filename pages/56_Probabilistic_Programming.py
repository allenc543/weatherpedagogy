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
df = load_data()
fdf = sidebar_filters(df)

chapter_header(56, "Probabilistic Programming", part="XIII")

st.markdown(
    "Last two chapters gave us beautiful closed-form Bayesian updates. Prior times "
    "likelihood, divide by evidence, done. But here is the catch: that only worked "
    "because we used a very specific combination -- a normal prior with a normal "
    "likelihood (the 'conjugate' case). The moment your model gets even slightly "
    "more realistic, the math breaks."
)
st.markdown(
    "**The problem**: We want to model the relationship between humidity and temperature "
    "in Dallas. Specifically: if I tell you the humidity is 45%, what is your best "
    "guess for the temperature, and how uncertain are you? This is Bayesian linear "
    "regression: temp = intercept + slope * humidity + noise. We want posterior "
    "distributions over the intercept, the slope, and the noise level."
)
st.markdown(
    "For this model, there is no closed-form posterior. The integral you need to compute "
    "does not have a nice analytical solution. So what do you do? You *approximate* it "
    "by generating samples from the posterior using a clever random walk algorithm called "
    "**Markov Chain Monte Carlo (MCMC)**. And the tools that automate this -- PyMC, Stan, "
    "Pyro -- are called **probabilistic programming languages**."
)
st.markdown(
    "**Why this matters**: Probabilistic programming lets you specify *any* Bayesian model "
    "in code and get posterior distributions automatically. Want to model temperature as "
    "a function of humidity, wind speed, and pressure simultaneously? Easy. Want a "
    "hierarchical model where each city has its own slope but they are all drawn from a "
    "shared group distribution? Also easy. The PPL handles the hard part (sampling from "
    "the posterior), and you focus on the modeling part (deciding what goes in)."
)

# ---------------------------------------------------------------------------
# 1. Theory
# ---------------------------------------------------------------------------
concept_box(
    "What Is Probabilistic Programming?",
    "In Chapters 54-55, we got nice closed-form posteriors because we used conjugate "
    "priors. That was lovely, but it only works for a tiny fraction of real models. "
    "The moment you want a linear regression (temp ~ humidity), a hierarchical model "
    "(different slopes per city), or anything with a non-Gaussian likelihood, the math "
    "stops being tractable.<br><br>"
    "Probabilistic programming languages (PPLs) like <b>PyMC</b>, Stan, and Pyro solve "
    "this by letting you specify your model in code, then automatically performing "
    "Bayesian inference using MCMC algorithms. The workflow is always the same:<br>"
    "1. Define priors for each parameter (intercept ~ N(20, 10), slope ~ N(0, 1), "
    "sigma ~ HalfNormal(5))<br>"
    "2. Define the likelihood (temp_observed ~ N(intercept + slope * humidity, sigma))<br>"
    "3. Let the PPL sample from the posterior automatically<br><br>"
    "You describe the generative story of your data (how you think the data was "
    "produced), and the PPL works backward to figure out what parameter values are "
    "consistent with what you actually observed. It is, frankly, a little magical.",
)

concept_box(
    "MCMC: Markov Chain Monte Carlo",
    "Here is the core idea. If you cannot compute the posterior analytically (and you "
    "usually cannot), you can instead generate <em>samples</em> from it by constructing "
    "a cleverly designed random walk through parameter space.<br><br>"
    "Imagine you are searching for the best combination of intercept and slope for our "
    "temp ~ humidity model. MCMC starts at some initial guess (say, intercept = 20, "
    "slope = -0.1). It proposes a small random step (maybe intercept = 20.05, slope = "
    "-0.098). If the new location has higher posterior probability (the data is more "
    "consistent with these parameters), it moves there. If lower, it moves there with "
    "some probability, or stays put. After thousands of steps, the chain has spent most "
    "of its time in high-probability regions of parameter space -- and the histogram of "
    "where the chain visited IS the posterior distribution.<br><br>"
    "Key diagnostics to watch for:<br>"
    "- <b>Trace plots</b>: should look like 'hairy caterpillars' (well-mixed, bouncing "
    "around a stable mean). If they look like a snake trying to escape a box (drifting, "
    "not mixing), the chain has not converged.<br>"
    "- <b>R-hat</b>: should be close to 1.0. Values above 1.05 mean your chains disagree "
    "with each other, which means the results are not trustworthy.<br>"
    "- <b>Effective sample size</b>: should be large (ideally > 400). Low ESS means "
    "high autocorrelation between samples.",
)

st.markdown("""
### The PyMC Workflow

Here is what a Bayesian linear regression for our weather data looks like in PyMC. Three blocks of code and you have posterior distributions for all parameters.

```python
import pymc as pm

with pm.Model() as model:
    # 1. Priors -- what do you believe before seeing data?
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    slope = pm.Normal('slope', mu=0, sigma=1)
    sigma = pm.HalfNormal('sigma', sigma=5)

    # 2. Likelihood -- how does humidity relate to temperature?
    mu = intercept + slope * x_data
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_data)

    # 3. Inference -- let PyMC figure out the posterior
    trace = pm.sample(1000, cores=2)
```

That is it. The PPL handles the MCMC sampling, adaptation, convergence checking, and diagnostics. You get posterior distributions over intercept, slope, and sigma -- not just point estimates, but full distributions that tell you exactly how uncertain you are about each parameter.
""")

warning_box(
    "PyMC requires a working installation with a compatible backend (JAX or C compiler). "
    "In this chapter, we simulate what PyMC would produce using analytical solutions and "
    "a Metropolis-Hastings sampler implemented in pure NumPy. Think of it as the "
    "pedagogical equivalent of a flight simulator -- same instruments, same principles, "
    "just without leaving the ground."
)

st.divider()

# ---------------------------------------------------------------------------
# 2. Interactive: Bayesian Linear Model (temp ~ humidity)
# ---------------------------------------------------------------------------
st.subheader("Interactive: Bayesian Linear Regression (Temperature ~ Humidity)")

st.markdown(
    "Here is the concrete model. We take a city's weather data and fit: "
    "**temperature = intercept + slope * humidity + noise**. The Bayesian version gives "
    "us not just one best-fit line, but a *distribution* of plausible lines. Each line "
    "has a different intercept and slope, drawn from the posterior distribution."
)
st.markdown(
    "**What the controls do:** 'City' selects the data. 'Data sample size' controls "
    "how many hourly readings we use. 'MCMC samples' controls how many posterior samples "
    "the Metropolis-Hastings algorithm generates. 'Prior std for slope' controls how "
    "informative the prior on the slope is -- a small value (0.1) says 'I am fairly "
    "confident the slope is near zero,' while a large value (5.0) says 'I have no idea.'"
)
st.markdown(
    "**What to look for:** The scatter plot shows the humidity-temperature relationship "
    "for your chosen city. Look for the overall trend: in most cities, higher humidity "
    "is weakly associated with lower temperatures (the slope is slightly negative). "
    "But the relationship is noisy -- humidity alone explains only a fraction of "
    "temperature variation."
)

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
    "Now we run the actual Bayesian inference. The Metropolis-Hastings algorithm will "
    "explore the space of all possible (intercept, slope, sigma) combinations, spending "
    "more time in regions that are consistent with our data and our priors. After "
    f"{n_mcmc} samples (plus 500 burn-in), we have an approximation of the posterior "
    "distribution."
)
st.markdown(
    "**The model being fitted:** temp = intercept + slope * humidity_centered + noise, "
    f"where humidity is centered (mean-subtracted and scaled). Using {n_sample} data "
    f"points from {pp_city}."
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

st.markdown(
    f"**Acceptance rate:** {acc_rate:.1%} (target: 20-50%). "
    f"{'Good -- the chain is exploring efficiently.' if 0.2 <= acc_rate <= 0.5 else 'Outside the ideal range -- the proposal step size may need tuning.'} "
    f"We generated {n_mcmc} posterior samples after discarding 500 burn-in samples."
)

# ---------------------------------------------------------------------------
# 4. Trace Plots
# ---------------------------------------------------------------------------
st.subheader("Trace Plots and Posterior Distributions")

st.markdown(
    "Trace plots are your MCMC diagnostic dashboard. The left column shows the chain's "
    "trajectory over time (should look like a 'hairy caterpillar' bouncing around a "
    "stable mean). The right column shows the histogram of posterior samples (the "
    "approximate posterior distribution). The dashed vertical lines mark the 95% "
    "credible interval."
)

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
    f"slope = {ols_params[1]:.4f}. The Bayesian posterior means should be very close "
    f"to these values -- with {n_sample} data points and weak priors, the data dominates "
    f"and the Bayesian answer converges to the frequentist one."
)

slope_lo, slope_hi = np.percentile(slope_original, [2.5, 97.5])
insight_box(
    f"The posterior mean slope is {np.mean(slope_original):.4f} -- meaning for each "
    f"1% increase in humidity, temperature changes by about {np.mean(slope_original):.3f} C. "
    f"The 95% credible interval for the slope is ({slope_lo:.4f}, {slope_hi:.4f}). "
    f"{'The interval excludes zero, so we are confident humidity and temperature are genuinely related.' if (slope_lo > 0 or slope_hi < 0) else 'The interval includes zero, so we cannot confidently say humidity affects temperature.'} "
    f"The Bayesian posterior means are very close to the OLS estimates, which is exactly "
    f"what you expect with weak priors and enough data. The key advantage is not a "
    f"different point estimate -- it is the full posterior distribution. You do not just "
    f"know the slope is around {np.mean(slope_original):.3f}; you know the probability it "
    f"is between {slope_lo:.3f} and {slope_hi:.3f}."
)

st.divider()

# ---------------------------------------------------------------------------
# 5. Posterior Predictive Plot
# ---------------------------------------------------------------------------
st.subheader("Posterior Predictive: Regression Lines from the Posterior")

st.markdown(
    "This is where the Bayesian approach really shines visually. Instead of drawing "
    "one best-fit line, we draw 100 lines -- each one a plausible regression line "
    "sampled from the posterior distribution. The spread of these lines IS your "
    "uncertainty about the humidity-temperature relationship."
)
st.markdown(
    "**What to look for:** Where the fan of red lines is narrow, the model is confident "
    "about the relationship. Where the fan is wide, the model is uncertain. The fan is "
    "typically narrowest near the center of the data (where we have the most observations) "
    "and widest at the extremes."
)

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
    "Each faint red line is one plausible regression line drawn from the posterior. "
    "The spread of these lines IS your uncertainty made visible. A frequentist analysis "
    "gives you one line and a confidence band; the Bayesian analysis gives you a "
    "distribution of lines, which is a more honest representation of what we know. "
    f"For {pp_city}, the lines are relatively tight around the mean (dark line), "
    f"indicating we are fairly confident about the humidity-temperature relationship. "
    f"The slope is {mean_slope:.4f} C per % humidity -- a weak but real signal buried "
    f"in a lot of noise (sigma = {np.mean(sigma_samples):.2f} C)."
)

st.divider()

# ---------------------------------------------------------------------------
# 6. Hierarchical Model Concept
# ---------------------------------------------------------------------------
st.subheader("Hierarchical Models: Partial Pooling Across Cities")

st.markdown(
    "Now let me show you the killer app of probabilistic programming: hierarchical models. "
    "We have temperature data from 6 cities. We want to estimate each city's mean "
    "temperature. There are three approaches:"
)
st.markdown(
    "**Complete pooling** ('all cities are the same'): Ignore city differences entirely, "
    "compute one global mean. This throws away real information -- NYC is obviously "
    "colder than Dallas.\n\n"
    "**No pooling** ('all cities are independent'): Compute each city's mean separately, "
    "as if the other cities do not exist. This ignores the fact that all 6 cities share "
    "the same planet and similar physics.\n\n"
    "**Partial pooling** ('cities are different, but related'): Each city has its own mean, "
    "but all 6 means are drawn from a shared group-level distribution. Cities with fewer "
    "observations get 'shrunk' toward the group mean -- they borrow strength from the "
    "other cities. This is the hierarchical model, and it is usually the best option."
)

concept_box(
    "Hierarchical (Multilevel) Models",
    "The hierarchical model says: each city's mean temperature mu_i is drawn from a "
    "shared group distribution N(mu_global, tau). Then each observation within that "
    "city is drawn from N(mu_i, sigma).<br><br>"
    "Concretely: there is some 'average city temperature' mu_global (maybe 20 C). "
    "Each city deviates from this: Dallas might be mu_global + 0.5, NYC might be "
    "mu_global - 7. The spread of these deviations is controlled by tau. And within "
    "each city, individual hourly readings vary with standard deviation sigma.<br><br>"
    "The magic of partial pooling: if a city has very few data points (say, we only "
    "have 10 readings for a new city), its estimate gets pulled strongly toward the "
    "group mean -- because with limited data, the model 'hedges' by borrowing information "
    "from the other 5 cities. A city with thousands of readings barely gets pulled at all. "
    "This automatic borrowing of strength is one of the most practically useful ideas in "
    "all of statistics.",
)

formula_box(
    "Hierarchical Model for City Temperatures",
    r"\underbrace{\mu_i}_{\text{city mean temp}} \sim \mathcal{N}(\underbrace{\mu_{\text{global}}}_{\text{group mean}}, \underbrace{\tau^2}_{\text{between-city variance}}), \quad "
    r"\underbrace{y_{ij}}_{\text{hourly reading}} \sim \mathcal{N}(\underbrace{\mu_i}_{\text{city mean}}, \underbrace{\sigma^2}_{\text{within-city variance}})",
    "mu_i is the mean temperature for city i; mu_global and tau describe the group-level "
    "distribution of city means; sigma is the within-city noise. The key insight: each "
    "city's mu_i is regularized toward mu_global, with the amount of regularization "
    "depending on how much data that city has.",
)

# Demonstrate shrinkage
st.markdown("#### Shrinkage Effect: Partial Pooling vs No Pooling")

st.markdown(
    "In the chart below, circles show each city's sample mean (no pooling), and diamonds "
    "show the partially-pooled estimate. Arrows show the direction and magnitude of "
    "shrinkage. Cities far from the grand mean get pulled inward; cities with less data "
    "get pulled more."
)

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

    st.markdown(
        "**Reading the table:** The 'Shrinkage Amount' column shows how much each city's "
        "estimate was pulled toward the grand mean. With ~17,500 observations per city "
        "(our dataset has 105,264 rows across 6 cities), the shrinkage is tiny -- a "
        "fraction of a degree. But in a scenario with uneven data (say, 10,000 readings "
        "for Dallas but only 50 for a new city), the new city's estimate would be shrunk "
        "substantially toward the group mean, which is almost always the right thing to do."
    )

    insight_box(
        "Partial pooling 'shrinks' each city's estimate toward the grand mean "
        f"({grand_mean:.1f} C). Cities with fewer observations are shrunk more -- they "
        "borrow more strength from the group. This is not a bug; it is the model saying: "
        "'I do not have much data for this city, so I will hedge toward what the other "
        "cities suggest.' With our dataset (roughly equal data per city), the shrinkage "
        "is small. But in real-world scenarios with unbalanced data, partial pooling "
        "consistently produces better out-of-sample predictions than either complete "
        "pooling or no pooling. It is the Goldilocks solution."
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
    explanation=(
        "MCMC generates samples from the posterior distribution by constructing a random "
        "walk (Markov chain) whose stationary distribution is the target posterior. In our "
        "weather example, each sample is a triplet (intercept, slope, sigma) -- one plausible "
        "set of parameter values. After 2,000 samples, the histogram of intercept values IS "
        "the posterior distribution of the intercept. You can compute any summary from these "
        "samples: posterior means, medians, credible intervals, probability that the slope is "
        "negative, etc. The key point: MCMC gives you the FULL posterior, not just a point "
        "estimate. That full distribution is what makes Bayesian inference powerful."
    ),
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
    explanation=(
        "Partial pooling means each city gets its own parameter (its own mean temperature), "
        "but all those parameters are linked through a shared group-level distribution. "
        "Dallas has its own mu_Dallas, NYC has its own mu_NYC, but both are drawn from "
        "N(mu_global, tau). This creates shrinkage: cities with less data are pulled more "
        "strongly toward the group mean. If we only have 50 readings for a new city, we "
        "do not trust them fully -- instead, we compromise between what those 50 readings "
        "say and what the other 5 cities suggest. This is a principled way to borrow strength "
        "across groups, and it almost always outperforms both 'ignore city differences' "
        "(complete pooling) and 'treat cities as independent' (no pooling)."
    ),
    key="ch56_quiz2",
)

quiz(
    "You run MCMC for a Bayesian regression and the trace plot shows the chain drifting "
    "steadily upward over 2,000 samples. What does this indicate?",
    [
        "The model is learning and improving over time",
        "The chain has not converged -- the posterior samples are not reliable",
        "The data is trending upward",
        "The prior is too strong",
    ],
    correct_idx=1,
    explanation=(
        "A drifting trace plot means the chain has NOT reached its stationary distribution "
        "-- it is still 'exploring' and has not settled into the high-probability region of "
        "parameter space. This is a convergence failure. The samples from a non-converged "
        "chain are NOT valid posterior samples. The fixes: (1) increase the burn-in period "
        "(discard more initial samples), (2) increase the total number of samples, (3) "
        "reparameterize the model to improve mixing, (4) check that the model is identified "
        "(the posterior actually has a peak). In PyMC, you would check R-hat > 1.05 or "
        "low effective sample size as diagnostic flags. Never trust MCMC results without "
        "checking convergence."
    ),
    key="ch56_quiz3",
)

st.divider()

# ---------------------------------------------------------------------------
# 9. Takeaways
# ---------------------------------------------------------------------------
takeaways([
    "**Probabilistic programming** automates Bayesian inference via MCMC for models "
    "where closed-form posteriors do not exist. You specify the model (temp = intercept "
    "+ slope * humidity + noise), and the PPL figures out the posterior.",
    "**MCMC** produces samples from the posterior -- each sample is one plausible set of "
    "parameter values. Check convergence with trace plots (should look like hairy "
    "caterpillars) and R-hat (should be < 1.05) before trusting anything.",
    "Bayesian regression gives you a **distribution of regression lines**, not just one "
    "best-fit line. The spread of those lines IS your uncertainty about the humidity-"
    "temperature relationship. Where the fan is narrow, you are confident; where it is "
    "wide, you are not.",
    "**Hierarchical models** enable partial pooling across cities: each city gets its "
    "own parameters, but cities with less data borrow strength from the group. This "
    "produces better out-of-sample predictions than treating cities independently or "
    "lumping them all together.",
    "With enough data and weak priors, Bayesian and frequentist estimates converge to "
    "the same answer. The Bayesian advantage is honest uncertainty quantification -- "
    "you get full posterior distributions, credible intervals, and the ability to answer "
    "questions like 'what is the probability that the slope is negative?'",
])

navigation(
    prev_label="Ch 55: Bayesian Inference",
    prev_page="55_Bayesian_Inference.py",
    next_label="Ch 57: Statistical Anomaly Detection",
    next_page="57_Statistical_Anomaly_Detection.py",
)
