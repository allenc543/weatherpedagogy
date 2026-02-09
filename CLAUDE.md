# Weather Data Science Pedagogy App

## Architecture

### Overview
A multi-page Streamlit app that teaches every major data science concept through interactive lessons using 2 years of hourly weather data from 6 US cities.

**Dataset**: `hourly_weather_data.csv` — ~105,264 rows, 6 cities (Dallas, San Antonio, Houston, Austin, NYC, LA), 4 numeric features (temperature_c, relative_humidity_pct, wind_speed_kmh, surface_pressure_hpa), hourly from 2024-02-10 to 2026-02-09.

### File Structure
```
weatherpedagogy/
├── app.py                          # Main entry point + welcome page
├── requirements.txt                # All dependencies
├── .streamlit/config.toml          # Theme + server config
├── hourly_weather_data.csv         # Data (fetched via fetch_weather.py)
├── fetch_weather.py                # Data fetcher (Open-Meteo API)
├── utils/
│   ├── __init__.py
│   ├── data_loader.py              # Cached data loading + filtering
│   ├── plotting.py                 # Shared Plotly helpers
│   ├── stats_helpers.py            # Reusable stats computations
│   ├── ml_helpers.py               # Model train/eval wrappers
│   ├── ui_components.py            # Concept boxes, quizzes, navigation
│   └── constants.py                # Colors, labels, city metadata
└── pages/                          # 63 chapter pages (Streamlit multipage)
    ├── 00_Welcome.py
    ├── 01_Exploring_the_Dataset.py
    ├── ... (numbered sequentially)
    └── 62_Capstone_Project.py
```

### Key Patterns
- **Streamlit multipage app**: Each `pages/XX_Name.py` is auto-discovered as a nav item
- **Caching**: `@st.cache_data` for data loading/computations, `@st.cache_resource` for fitted models
- **Session state**: Filter persistence, quiz scores, progress tracking
- **Page template**: Every chapter follows: Header → Theory → Interactive Demo → Code Example → Quiz → Takeaways → Nav
- **Sidebar**: City multi-select and date range filters shared across all pages via `data_loader.load_data()` and `data_loader.apply_filters()`

### Chapter Curriculum (62 Chapters, 15 Parts)

| Part | Chapters | Topics |
|------|----------|--------|
| I: Foundations | 1-4 | DataFrames, descriptive stats, distributions, sampling |
| II: Visualization | 5-10 | Histograms, scatter, time series, box plots, heatmaps, advanced |
| III: Statistical Inference | 11-15 | CIs, hypothesis tests, A/B, ANOVA, nonparametric |
| IV: Correlation & Regression | 16-20 | Correlation, simple/multiple/polynomial regression, regularization |
| V: Classification | 21-26 | Logistic regression, trees, RF, SVM, KNN, Naive Bayes |
| VI: Clustering | 27-30 | K-Means, hierarchical, DBSCAN, silhouette |
| VII: Dimensionality Reduction | 31-33 | PCA, t-SNE, UMAP |
| VIII: Time Series | 34-38 | Decomposition, ACF, ARIMA, Prophet, seasonality |
| IX: Feature Engineering | 39-42 | Feature creation, scaling, selection, importance |
| X: Model Evaluation | 43-46 | CV, bias-variance, ROC/AUC, regression metrics |
| XI: Ensemble Methods | 47-49 | Bagging, boosting, stacking |
| XII: Deep Learning | 50-53 | Neural nets, feedforward, RNN/LSTM, autoencoders |
| XIII: Bayesian Methods | 54-56 | Bayes theorem, inference, probabilistic programming |
| XIV: Anomaly Detection | 57-59 | Statistical, isolation forest, autoencoder |
| XV: Causal Inference | 60-61 | Correlation vs causation, natural experiments |
| Capstone | 62 | End-to-end project |


### Tech Stack
- **Frontend**: Streamlit
- **Plotting**: Plotly (primary), Matplotlib/Seaborn (when needed)
- **Stats**: scipy, statsmodels, pingouin
- **ML**: scikit-learn, xgboost, lightgbm
- **Deep Learning**: PyTorch
- **Time Series**: statsmodels, pmdarima, prophet
- **Bayesian**: PyMC, ArviZ
- **Dim Reduction**: scikit-learn (PCA), umap-learn
- **Explainability**: SHAP

### Development Notes
- Run with: `streamlit run app.py`
- Data fetch: `python fetch_weather.py` (uses free Open-Meteo API)
- All utility functions use `@st.cache_data` or `@st.cache_resource` for performance
- City colors and metadata defined in `utils/constants.py`
