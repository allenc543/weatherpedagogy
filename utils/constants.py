"""Shared constants: colors, labels, city metadata."""

CITY_COLORS = {
    "Dallas": "#E63946",
    "San Antonio": "#F4A261",
    "Houston": "#2A9D8F",
    "Austin": "#264653",
    "NYC": "#7209B7",
    "Los Angeles": "#FB8500",
}

CITY_LIST = list(CITY_COLORS.keys())

TEXAS_CITIES = ["Dallas", "San Antonio", "Houston", "Austin"]
COASTAL_CITIES = ["Houston", "Los Angeles"]
INLAND_CITIES = ["Dallas", "Austin", "San Antonio"]

FEATURE_COLS = ["temperature_c", "relative_humidity_pct", "wind_speed_kmh", "surface_pressure_hpa"]

FEATURE_LABELS = {
    "temperature_c": "Temperature (°C)",
    "relative_humidity_pct": "Relative Humidity (%)",
    "wind_speed_kmh": "Wind Speed (km/h)",
    "surface_pressure_hpa": "Surface Pressure (hPa)",
}

FEATURE_UNITS = {
    "temperature_c": "°C",
    "relative_humidity_pct": "%",
    "wind_speed_kmh": "km/h",
    "surface_pressure_hpa": "hPa",
}

SEASONS = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}

SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]

PART_TITLES = {
    "I": "Foundations",
    "II": "Visualization",
    "III": "Statistical Inference",
    "IV": "Correlation & Regression",
    "V": "Classification",
    "VI": "Clustering",
    "VII": "Dimensionality Reduction",
    "VIII": "Time Series",
    "IX": "Feature Engineering",
    "X": "Model Evaluation",
    "XI": "Ensemble Methods",
    "XII": "Deep Learning",
    "XIII": "Bayesian Methods",
    "XIV": "Anomaly Detection",
    "XV": "Causal Inference",
}
