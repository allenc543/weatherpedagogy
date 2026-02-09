"""Weather Data Science Pedagogy App — Main Entry Point."""
import streamlit as st

st.set_page_config(
    page_title="Weather Data Science Pedagogy",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Weather Data Science Pedagogy")
st.subheader("A somewhat obsessive attempt to teach every major data science concept through the medium of weather")

st.markdown("""
Here is a thing I believe: the best way to learn data science is not through toy datasets about
iris flowers or house prices, but through data you can *feel*. Everyone has opinions about weather.
Everyone has stood outside and thought "this doesn't seem right." That intuition -- the gut sense
that today is weirdly warm for January -- is, it turns out, the same intuition that powers
Bayesian reasoning, anomaly detection, and half of modern statistics.

So I built this. **62 chapters** of interactive data science, all grounded in real weather data,
all designed to connect what the math says to what you already know about stepping outside.

### The Dataset

We have **105,264 hourly observations** from February 2024 to February 2026, covering
**6 US cities**: Dallas, San Antonio, Houston, Austin, NYC, and Los Angeles. Four measurements
per reading: temperature, relative humidity, wind speed, and surface pressure.

This is enough data to do real statistics but not so much that your browser will catch fire.
A deliberate design choice.

### How to Use This App

1. **Navigate** via the sidebar to any chapter. They build on each other, but you can jump around
   if you are the kind of person who reads the last page of a novel first (no judgment, mostly)
2. **Filter** data by city and date range -- every visualization updates in real time
3. **Interact** with sliders, dropdowns, and toggles. The best way to build intuition is to
   break things and see what happens
4. **Test yourself** with quizzes, because retrieval practice works even when you know it works

### Course Outline
""")

parts = {
    "Part I: Foundations (Ch 1-4)": "DataFrames, descriptive stats, probability distributions, sampling",
    "Part II: Visualization (Ch 5-10)": "Histograms, scatter plots, time series, box plots, heatmaps",
    "Part III: Statistical Inference (Ch 11-15)": "Confidence intervals, hypothesis testing, ANOVA",
    "Part IV: Correlation & Regression (Ch 16-20)": "Correlation, linear/polynomial regression, regularization",
    "Part V: Classification (Ch 21-26)": "Logistic regression, decision trees, random forests, SVM, KNN",
    "Part VI: Clustering (Ch 27-30)": "K-Means, hierarchical, DBSCAN, silhouette analysis",
    "Part VII: Dimensionality Reduction (Ch 31-33)": "PCA, t-SNE, UMAP",
    "Part VIII: Time Series (Ch 34-38)": "Decomposition, ACF, ARIMA, Prophet, seasonality",
    "Part IX: Feature Engineering (Ch 39-42)": "Feature creation, scaling, selection, importance",
    "Part X: Model Evaluation (Ch 43-46)": "Cross-validation, bias-variance, ROC/AUC, regression metrics",
    "Part XI: Ensemble Methods (Ch 47-49)": "Bagging, boosting, stacking",
    "Part XII: Deep Learning (Ch 50-53)": "Neural nets, feedforward, LSTM, autoencoders",
    "Part XIII: Bayesian Methods (Ch 54-56)": "Bayes theorem, inference, probabilistic programming",
    "Part XIV: Anomaly Detection (Ch 57-59)": "Statistical, isolation forest, autoencoder",
    "Part XV: Causal Inference (Ch 60-61)": "Correlation vs causation, natural experiments",
    "Capstone (Ch 62)": "End-to-end data science project",
}

for part, desc in parts.items():
    st.markdown(f"**{part}** -- {desc}")

st.divider()
st.markdown("**Pick a chapter from the sidebar and let's get started. The weather isn't going to analyze itself.**")

# Show dataset preview
st.subheader("Dataset Preview")
from utils.data_loader import load_data
df = load_data()
st.dataframe(df.head(20), use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Rows", f"{len(df):,}")
col2.metric("Cities", df["city"].nunique())
col3.metric("Date Range", f"{df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}")
col4.metric("Features", "4 numeric")
