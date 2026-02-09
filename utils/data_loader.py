"""Cached data loading and filtering utilities."""
import streamlit as st
import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hourly_weather_data.csv")


@st.cache_data
def load_data():
    """Load the full weather dataset with datetime parsing and derived columns."""
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["year"] = df["datetime"].dt.year
    df["date"] = df["datetime"].dt.date
    df["month_name"] = df["datetime"].dt.month_name()
    from utils.constants import SEASONS
    df["season"] = df["month"].map(SEASONS)
    return df


def sidebar_filters(df):
    """Render sidebar city and date filters; return filtered DataFrame."""
    from utils.constants import CITY_LIST
    st.sidebar.header("Filters")
    if "selected_cities" not in st.session_state:
        st.session_state.selected_cities = CITY_LIST.copy()
    selected = st.sidebar.multiselect(
        "Cities", CITY_LIST,
        default=st.session_state.selected_cities,
        key="city_filter"
    )
    st.session_state.selected_cities = selected

    min_date = df["datetime"].min().date()
    max_date = df["datetime"].max().date()
    date_range = st.sidebar.date_input(
        "Date range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date,
        key="date_filter"
    )
    if len(date_range) == 2:
        start, end = date_range
    else:
        start, end = min_date, max_date

    mask = (
        df["city"].isin(selected) &
        (df["datetime"].dt.date >= start) &
        (df["datetime"].dt.date <= end)
    )
    return df[mask].copy()


def get_city_data(df, city):
    """Filter DataFrame to a single city."""
    return df[df["city"] == city].copy()
