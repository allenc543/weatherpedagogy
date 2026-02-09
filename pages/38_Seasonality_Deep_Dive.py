"""Chapter 38 -- Seasonality Deep Dive: Diurnal, Annual, and Fourier Analysis."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(38, "Seasonality Deep Dive", part="VIII")

st.markdown(
    "Weather data has a nesting-doll structure of cycles within cycles. There is "
    "a **24-hour diurnal cycle** (the sun comes up, things get warm; the sun goes "
    "down, things cool off) sitting inside a **365-day annual cycle** (the Earth "
    "tilts toward the sun in summer, away in winter). Both of these are approximately "
    "sinusoidal, which means they are perfect candidates for Fourier analysis -- the "
    "mathematical technique of decomposing any signal into a sum of sine and cosine "
    "waves. This chapter is where we put on our signal-processing hats and ask: "
    "what frequencies are hiding in this data?"
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

concept_box(
    "Multiple Seasonalities in Weather",
    "<b>Diurnal cycle (24 h)</b>: Temperature rises after sunrise, peaks in "
    "early-to-mid afternoon (not at solar noon -- there is a lag!), and falls "
    "overnight. This is driven by the time it takes the ground to absorb and "
    "re-radiate solar energy.<br>"
    "<b>Annual cycle (365 d)</b>: Warm summers and cold winters, driven by "
    "Earth's 23.5-degree axial tilt. The tilt changes how much solar energy "
    "each hemisphere receives, not the distance from the sun (a common "
    "misconception).<br>"
    "Both cycles are approximately sinusoidal, which is why Fourier analysis "
    "works so well on weather data."
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.subheader("Seasonality Settings")
city = st.sidebar.selectbox("City", CITY_LIST, key="season_city")
city_df = fdf[fdf["city"] == city].copy()

# ── Section 1: Diurnal Cycle (24-Hour) ───────────────────────────────────────
st.header("1. Diurnal Cycle (24-Hour Pattern)")

st.markdown(
    "We average temperature at each hour across all days to reveal the typical "
    "daily profile. This is the diurnal cycle stripped of noise -- what the "
    "average day looks like in this city."
)

hourly_avg = city_df.groupby("hour")["temperature_c"].agg(["mean", "std"]).reset_index()

fig_diurnal = go.Figure()
fig_diurnal.add_trace(go.Scatter(
    x=hourly_avg["hour"], y=hourly_avg["mean"],
    mode="lines+markers", name="Mean Temperature",
    line=dict(color=CITY_COLORS.get(city, "#2E86C1"), width=3),
))
# Shade +/- 1 std
fig_diurnal.add_trace(go.Scatter(
    x=list(hourly_avg["hour"]) + list(hourly_avg["hour"][::-1]),
    y=list(hourly_avg["mean"] + hourly_avg["std"])
      + list((hourly_avg["mean"] - hourly_avg["std"])[::-1]),
    fill="toself", fillcolor="rgba(46,134,193,0.15)",
    line=dict(width=0), name="+/- 1 Std Dev",
))
fig_diurnal.update_layout(
    xaxis_title="Hour of Day", yaxis_title="Temperature (deg C)",
    xaxis=dict(dtick=2),
)
apply_common_layout(fig_diurnal, f"Average Diurnal Temperature Profile: {city}", 400)
st.plotly_chart(fig_diurnal, use_container_width=True)

# Diurnal amplitude
diurnal_amp = hourly_avg["mean"].max() - hourly_avg["mean"].min()
peak_hour = int(hourly_avg.loc[hourly_avg["mean"].idxmax(), "hour"])
trough_hour = int(hourly_avg.loc[hourly_avg["mean"].idxmin(), "hour"])

c1, c2, c3 = st.columns(3)
c1.metric("Diurnal Amplitude", f"{diurnal_amp:.1f} deg C")
c2.metric("Warmest Hour", f"{peak_hour}:00")
c3.metric("Coolest Hour", f"{trough_hour}:00")

insight_box(
    f"In {city}, the typical diurnal swing is **{diurnal_amp:.1f} deg C**. "
    f"Temperature peaks around {peak_hour}:00 and bottoms out near {trough_hour}:00. "
    "Notice that the warmest hour is *not* noon, when solar radiation is strongest -- "
    "it is a few hours later. This 'thermal lag' exists because the ground needs time "
    "to absorb solar energy and re-radiate it as heat. It is the same reason a frying "
    "pan stays hot after you turn off the stove."
)

# All cities comparison
st.subheader("Diurnal Profile -- All Cities")
all_hourly = fdf.groupby(["hour", "city"])["temperature_c"].mean().reset_index()
fig_all_d = px.line(
    all_hourly, x="hour", y="temperature_c", color="city",
    color_discrete_map=CITY_COLORS,
    labels={"temperature_c": "Temperature (deg C)", "hour": "Hour"},
    title="Average Diurnal Temperature by City",
)
apply_common_layout(fig_all_d, height=450)
st.plotly_chart(fig_all_d, use_container_width=True)

# ── Section 2: Annual Cycle (365-Day) ────────────────────────────────────────
st.header("2. Annual Cycle (365-Day Pattern)")

st.markdown(
    "Same trick, different scale. We average temperature by day-of-year across "
    "all years in the dataset. The result is the annual seasonal profile -- the "
    "shape that decomposition methods and Fourier analysis are trying to capture."
)

annual_avg = city_df.groupby("day_of_year")["temperature_c"].agg(["mean", "std"]).reset_index()

fig_annual = go.Figure()
fig_annual.add_trace(go.Scatter(
    x=annual_avg["day_of_year"], y=annual_avg["mean"],
    mode="lines", name="Mean Temperature",
    line=dict(color=CITY_COLORS.get(city, "#2E86C1"), width=2),
))
fig_annual.add_trace(go.Scatter(
    x=list(annual_avg["day_of_year"]) + list(annual_avg["day_of_year"][::-1]),
    y=list(annual_avg["mean"] + annual_avg["std"])
      + list((annual_avg["mean"] - annual_avg["std"])[::-1]),
    fill="toself", fillcolor="rgba(46,134,193,0.15)",
    line=dict(width=0), name="+/- 1 Std Dev",
))
fig_annual.update_layout(
    xaxis_title="Day of Year", yaxis_title="Temperature (deg C)",
)
apply_common_layout(fig_annual, f"Annual Temperature Profile: {city}", 400)
st.plotly_chart(fig_annual, use_container_width=True)

annual_amp = annual_avg["mean"].max() - annual_avg["mean"].min()
c1, c2 = st.columns(2)
c1.metric("Annual Amplitude", f"{annual_amp:.1f} deg C")
c2.metric("Range (mean)", f"{annual_avg['mean'].min():.1f} to {annual_avg['mean'].max():.1f} deg C")

# All cities annual
st.subheader("Annual Profile -- All Cities")
all_annual = fdf.groupby(["day_of_year", "city"])["temperature_c"].mean().reset_index()
fig_all_a = px.line(
    all_annual, x="day_of_year", y="temperature_c", color="city",
    color_discrete_map=CITY_COLORS,
    labels={"temperature_c": "Temperature (deg C)", "day_of_year": "Day of Year"},
    title="Annual Temperature by City",
)
apply_common_layout(fig_all_a, height=450)
st.plotly_chart(fig_all_a, use_container_width=True)

insight_box(
    "LA has the smallest annual amplitude because the Pacific Ocean acts as a "
    "giant thermal buffer -- water heats up and cools down much more slowly than "
    "land. NYC and Dallas have the largest swings because they are dominated by "
    "continental climate effects, where there is no ocean to smooth things out. "
    "This is not just trivia -- it directly affects what forecasting models need "
    "to capture."
)

# ── Section 3: Fourier Decomposition ─────────────────────────────────────────
st.header("3. Fourier Decomposition of Temperature")

concept_box(
    "Fourier Analysis",
    "Here is one of the most beautiful ideas in all of mathematics: any periodic "
    "signal -- any signal that repeats -- can be represented as a sum of sine and "
    "cosine waves at different frequencies. A square wave? Sum of sines. A sawtooth? "
    "Sum of sines. Temperature? Also a sum of sines. By computing the Fourier "
    "transform, we can identify exactly *which* frequencies are present and how "
    "strong each one is. The dominant frequency in our temperature data will be -- "
    "spoiler alert -- one cycle per 365 days. This is what scientists call 'seasons.'"
)

formula_box(
    "Discrete Fourier Transform",
    r"X_k = \sum_{n=0}^{N-1} x_n \, e^{-i 2\pi k n / N}",
    "X_k gives the amplitude and phase at frequency k/N. The amplitude tells you "
    "'how strong is this frequency?'; the phase tells you 'when does the cycle peak?'"
)

st.sidebar.subheader("Fourier Settings")
n_harmonics = st.sidebar.slider("Number of Fourier Harmonics", 1, 20, 5, key="fourier_n")

# Use daily mean for cleaner Fourier analysis
daily_city = city_df.groupby("date")["temperature_c"].mean().reset_index().sort_values("date")
daily_city["date"] = pd.to_datetime(daily_city["date"])
signal = daily_city["temperature_c"].values

# Remove mean (center)
signal_centered = signal - np.mean(signal)
N = len(signal_centered)

# FFT
fft_vals = np.fft.rfft(signal_centered)
freqs = np.fft.rfftfreq(N, d=1)  # cycles per day
periods = np.where(freqs > 0, 1 / freqs, np.inf)
magnitudes = 2 * np.abs(fft_vals) / N

# Period spectrum
spec_df = pd.DataFrame({
    "Period (days)": periods[1:100],
    "Magnitude": magnitudes[1:100],
})

fig_spec = go.Figure()
fig_spec.add_trace(go.Scatter(
    x=spec_df["Period (days)"], y=spec_df["Magnitude"],
    mode="lines", line=dict(color="#264653"),
))
fig_spec.update_layout(
    xaxis_title="Period (days)", yaxis_title="Amplitude (deg C)",
    xaxis_type="log",
)
apply_common_layout(fig_spec, f"Fourier Spectrum: {city}", 400)
st.plotly_chart(fig_spec, use_container_width=True)

# Find dominant period
top_idx = np.argsort(magnitudes[1:])[-3:] + 1
st.markdown("**Top 3 dominant periods:**")
for idx in reversed(top_idx):
    if freqs[idx] > 0:
        st.markdown(f"- Period = **{1/freqs[idx]:.1f} days** (amplitude = {magnitudes[idx]:.2f} deg C)")

insight_box(
    "Our Fourier decomposition found that the temperature signal contains a strong "
    "cycle near **365 days**. This is what scientists call 'seasons.' (I am being "
    "slightly cheeky, but the point is serious: Fourier analysis *independently "
    "discovers* the annual cycle from the data, with no prior knowledge of astronomy.) "
    "Smaller peaks may correspond to half-year harmonics or other subtle patterns."
)

# ── Section 4: Fourier Reconstruction ────────────────────────────────────────
st.header("4. Fourier Reconstruction")

st.markdown(
    f"Here is the truly wild part. We reconstruct the entire temperature signal "
    f"using only the top **{n_harmonics} Fourier harmonics**. That is, we take just "
    f"{n_harmonics} sine/cosine waves and add them together. The question is: how "
    "close do we get to the actual temperature pattern with so few ingredients?"
)

# Reconstruct with top harmonics
fft_filtered = np.zeros_like(fft_vals)
top_indices = np.argsort(np.abs(fft_vals[1:]))[-n_harmonics:] + 1
fft_filtered[top_indices] = fft_vals[top_indices]
reconstructed = np.fft.irfft(fft_filtered, n=N) + np.mean(signal)

fig_recon = go.Figure()
fig_recon.add_trace(go.Scatter(
    x=daily_city["date"], y=signal,
    mode="lines", name="Actual", line=dict(color="#264653", width=1),
    opacity=0.6,
))
fig_recon.add_trace(go.Scatter(
    x=daily_city["date"], y=reconstructed,
    mode="lines", name=f"Fourier ({n_harmonics} harmonics)",
    line=dict(color="#E63946", width=2),
))
apply_common_layout(fig_recon, f"Fourier Reconstruction: {city}", 450)
st.plotly_chart(fig_recon, use_container_width=True)

residual_std = np.std(signal - reconstructed)
st.metric(
    "Residual Std Dev",
    f"{residual_std:.2f} deg C",
    help="Standard deviation of the difference between actual and reconstructed.",
)

warning_box(
    "Fourier analysis assumes a stationary periodic signal -- meaning the cycles "
    "do not change over time and there is no trend. If your data has a trend, "
    "detrend it first. Also, Fourier works best with long, complete time series "
    "(multiple full cycles). A single year of data cannot reliably resolve a "
    "365-day period."
)

# ── Section 5: Seasonal Naive Baseline ───────────────────────────────────────
st.header("5. Seasonal Naive Baseline")

st.markdown(
    "Before you get too excited about fancy models, let me introduce the humbling "
    "**seasonal naive** forecast. It predicts that today's temperature equals the "
    "temperature *exactly one year ago*. That is the whole model. No parameters. "
    "No training. No Fourier terms. Just 'look at last year.' And it is a "
    "surprisingly strong baseline -- so strong that if your elaborate model cannot "
    "beat it, your elaborate model is not adding value."
)

# Build a naive forecast: lag by 365 days
daily_s = daily_city.set_index("date").sort_index()
daily_s["naive_365"] = daily_s["temperature_c"].shift(365)
daily_s = daily_s.dropna()

if len(daily_s) > 0:
    from sklearn.metrics import mean_absolute_error
    naive_mae = mean_absolute_error(daily_s["temperature_c"], daily_s["naive_365"])

    fig_naive = go.Figure()
    fig_naive.add_trace(go.Scatter(
        x=daily_s.index, y=daily_s["temperature_c"],
        mode="lines", name="Actual", line=dict(color="#264653"),
    ))
    fig_naive.add_trace(go.Scatter(
        x=daily_s.index, y=daily_s["naive_365"],
        mode="lines", name="Seasonal Naive (lag 365d)",
        line=dict(color="#F4A261", dash="dash"),
    ))
    apply_common_layout(fig_naive, "Seasonal Naive Forecast", 400)
    st.plotly_chart(fig_naive, use_container_width=True)

    st.metric("Seasonal Naive MAE", f"{naive_mae:.2f} deg C")

    insight_box(
        f"The seasonal naive baseline achieves MAE = {naive_mae:.2f} deg C. "
        "This is the number every serious forecasting model needs to beat. "
        "If your deep learning model with 47 layers and a GPU cluster cannot do "
        "better than 'use last year,' something has gone wrong. Always, always, "
        "always compare against baselines."
    )

code_example("""
import numpy as np

# FFT of daily mean temperature
signal = daily_temp.values
signal_centered = signal - np.mean(signal)
fft_vals = np.fft.rfft(signal_centered)
freqs = np.fft.rfftfreq(len(signal), d=1)  # cycles per day

# Dominant period
magnitudes = 2 * np.abs(fft_vals) / len(signal)
dominant_freq = freqs[np.argmax(magnitudes[1:]) + 1]
dominant_period = 1 / dominant_freq
print(f"Dominant period: {dominant_period:.0f} days")

# Seasonal naive forecast
forecast = series.shift(365)
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "In hourly weather data, what two dominant periodicities would you expect "
    "to see in a Fourier spectrum?",
    [
        "7 days and 30 days",
        "24 hours and 365 days",
        "12 hours and 52 weeks",
        "1 hour and 1 year",
    ],
    1,
    "The diurnal cycle (24 hours) and the annual cycle (365 days) are the two "
    "primary periodicities in weather data. They correspond to the two dominant "
    "astronomical cycles that drive weather: Earth's rotation and its orbit.",
    key="season_quiz",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "Weather has two dominant cycles: diurnal (24 h) and annual (365 d). Everything else is noise, trend, or subtle higher-order harmonics.",
    "The diurnal cycle is driven by solar heating; peak temperature lags solar noon by 2-3 hours because thermal inertia is a thing.",
    "Fourier analysis can independently discover these cycles from raw data -- no astronomy textbook required. It decomposes any periodic signal into sine/cosine components.",
    "A few Fourier harmonics can approximate the annual temperature pattern remarkably well. With 5 harmonics, you capture most of the seasonal shape.",
    "The seasonal naive baseline (lag 365 days) is a strong benchmark. Any model that cannot beat it is not earning its complexity.",
])
