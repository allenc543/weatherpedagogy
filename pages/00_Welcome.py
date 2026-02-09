"""Welcome & Course Overview page."""
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.plotting import line_chart, apply_common_layout, color_map
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import concept_box, insight_box, navigation

# ── Page config ──────────────────────────────────────────────────────────────
st.title("Welcome to Weather Data Science Pedagogy")
st.markdown(
    "Here's the pitch: you're going to learn every major data-science concept, "
    "and you're going to learn it by playing with **real hourly weather data** "
    "from six US cities. No toy datasets. No hand-waving. Just weather."
)
st.divider()

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()

# ── Course overview ──────────────────────────────────────────────────────────
st.header("Course Overview")
concept_box(
    "So What Is This, Exactly?",
    "This is an interactive course that teaches data science from the ground up, "
    "using two years of hourly weather observations from Dallas, San Antonio, Houston, "
    "Austin, New York City, and Los Angeles. Every single concept gets illustrated "
    "with real data you can explore, filter, and visualize yourself. No trust required -- "
    "you can verify everything as we go."
)

st.subheader("What You Will Learn")
topics = {
    "Foundations (Ch 1-4)": "DataFrames, descriptive statistics, probability distributions, sampling & estimation",
    "Visualization (Ch 5-10)": "Histograms, scatter plots, time series, box plots, heatmaps, advanced charts",
}
for part, desc in topics.items():
    st.markdown(f"**{part}** -- {desc}")

# ── Dataset summary ──────────────────────────────────────────────────────────
st.header("Dataset at a Glance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Rows", f"{len(df):,}")
col2.metric("Cities", df["city"].nunique())
date_min = df["datetime"].min().strftime("%Y-%m-%d")
date_max = df["datetime"].max().strftime("%Y-%m-%d")
col3.metric("Date Range", f"{date_min} to {date_max}")
col4.metric("Numeric Features", len(FEATURE_COLS))

st.subheader("Summary Statistics by City")
summary = (
    df.groupby("city")[FEATURE_COLS]
    .mean()
    .round(2)
    .rename(columns=FEATURE_LABELS)
)
st.dataframe(summary, use_container_width=True)

# ── Preview visualization ────────────────────────────────────────────────────
st.header("Preview: Temperature Over Time")

st.markdown(
    "Before we get into any theory, let's just *look* at the data. The chart below "
    "shows daily average temperature for every city in the dataset. Hover over any "
    "line to see exact values. Already you can start noticing things -- and that noticing "
    "is where data science begins."
)

daily_temp = (
    df.groupby(["date", "city"])["temperature_c"]
    .mean()
    .reset_index()
)
daily_temp["date"] = pd.to_datetime(daily_temp["date"])

fig = px.line(
    daily_temp,
    x="date",
    y="temperature_c",
    color="city",
    color_discrete_map=CITY_COLORS,
    labels={"temperature_c": "Temperature (°C)", "date": "Date", "city": "City"},
    title="Daily Average Temperature by City",
)
apply_common_layout(fig, height=500)
st.plotly_chart(fig, use_container_width=True)

insight_box(
    "Here's something to chew on before we even start the course: look at how Los Angeles "
    "barely moves compared to Dallas or NYC. LA lives in this narrow comfortable band "
    "while Dallas swings wildly between extremes. We'll put precise numbers on this "
    "difference in Chapter 2 (Descriptive Statistics), and when we do, you'll see that "
    "'standard deviation' isn't just a formula -- it's capturing something you can already see."
)

# ── Quick data preview ───────────────────────────────────────────────────────
st.header("Raw Data Preview")
preview_rows = st.slider("Number of rows to preview", 5, 100, 20, key="welcome_preview")
st.dataframe(df.head(preview_rows), use_container_width=True)

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    next_label="Ch 1: Exploring the Dataset",
    next_page="01_Exploring_the_Dataset.py",
)
