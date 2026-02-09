"""Chapter 1: Exploring the Dataset -- DataFrames, dtypes, shape, describe(), missing values."""
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout, color_map
from utils.constants import CITY_COLORS, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(1, "Exploring the Dataset", part="I")
st.markdown(
    "Before you can do anything interesting with data, you have to *look at it*. "
    "This sounds obvious, but you'd be amazed how many analyses go wrong because "
    "someone skipped this step. In this chapter, we'll learn how to inspect a "
    "pandas DataFrame -- its shape, data types, summary statistics, and missing "
    "values -- because **understanding your raw material is the first real skill**."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── Theory: What is a DataFrame? ─────────────────────────────────────────────
st.header("1.1  What Is a DataFrame?")
concept_box(
    "The DataFrame: Your New Best Friend",
    "A <b>DataFrame</b> is basically a spreadsheet that lives inside Python. "
    "It's two-dimensional (rows and columns), each column can have its own data type "
    "(numbers, text, dates, whatever), and it comes with an absurd number of built-in "
    "methods for slicing, dicing, and summarizing data. Rows are observations -- "
    "in our case, one hour of weather at one city. Columns are variables -- "
    "temperature, wind speed, humidity, and so on."
)

# ── df.head() ────────────────────────────────────────────────────────────────
st.header("1.2  Previewing the Data -- `df.head()`")
n_rows = st.slider("Rows to preview", 5, 50, 10, key="head_slider")
st.dataframe(fdf.head(n_rows), use_container_width=True)

code_example(
    """import pandas as pd

df = pd.read_csv("hourly_weather_data.csv", parse_dates=["datetime"])
df.head(10)
"""
)

# ── Shape & dtypes ───────────────────────────────────────────────────────────
st.header("1.3  Shape & Data Types")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Shape")
    st.write(f"Rows: **{fdf.shape[0]:,}**  |  Columns: **{fdf.shape[1]}**")

with col2:
    st.subheader("Data Types")
    dtype_df = pd.DataFrame({
        "Column": fdf.dtypes.index,
        "Dtype": fdf.dtypes.astype(str).values,
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

concept_box(
    "Why Should You Care About dtypes?",
    "Here's a mistake that has ruined many an afternoon: you try to compute an average "
    "and get a cryptic error because pandas thinks your numbers are strings. "
    "Numeric columns (float64, int64) support math. Object/string columns are for categories. "
    "Datetime columns let you do time-based magic like resampling and rolling windows. "
    "Knowing your dtypes upfront saves you from debugging something that shouldn't have been a bug."
)

code_example(
    """print(df.shape)      # (rows, columns)
print(df.dtypes)     # dtype of each column
print(df.info())     # concise summary including memory usage
"""
)

# ── df.describe() ────────────────────────────────────────────────────────────
st.header("1.4  Summary Statistics -- `df.describe()`")

st.markdown("Pick which columns you want to summarize. Go ahead, try different combinations.")
selected_cols = st.multiselect(
    "Columns",
    options=FEATURE_COLS,
    default=FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    key="describe_cols",
)

if selected_cols:
    desc = fdf[selected_cols].describe().T
    desc = desc.rename(index=FEATURE_LABELS).round(2)
    st.dataframe(desc, use_container_width=True)
else:
    st.info("Please select at least one column.")

insight_box(
    "`describe()` is the Swiss Army knife of first-look analysis. One call and you get "
    "count, mean, standard deviation, min, 25th percentile, median, 75th percentile, "
    "and max. Think of it as a quick health check -- if the min temperature is -900, "
    "you know something went wrong in data collection."
)

code_example(
    """df[["temperature_c", "wind_speed_kmh"]].describe()
"""
)

# ── Per-city describe ────────────────────────────────────────────────────────
st.header("1.5  Per-City Summary")
feature_choice = st.selectbox(
    "Feature to examine by city",
    FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
    key="percity_feat",
)
per_city = fdf.groupby("city")[feature_choice].describe().round(2)
st.dataframe(per_city, use_container_width=True)

# ── Missing values ───────────────────────────────────────────────────────────
st.header("1.6  Missing-Value Analysis")

concept_box(
    "The Missing Data Problem",
    "Real-world data is messy. Sensors go offline. Records get corrupted. Someone "
    "accidentally unplugs a weather station. The result: missing values (NaN). "
    "Before you do any modeling, you have to decide what to do about them. "
    "Your options: <b>drop</b> the incomplete rows, <b>fill</b> them with something "
    "reasonable (mean, median, forward-fill from the previous value), or <b>flag</b> "
    "them as a separate indicator so your model knows the data was missing."
)

missing = fdf.isnull().sum().reset_index()
missing.columns = ["Column", "Missing Count"]
missing["Missing %"] = (missing["Missing Count"] / len(fdf) * 100).round(2)
st.dataframe(missing, use_container_width=True, hide_index=True)

total_missing = fdf.isnull().sum().sum()
if total_missing == 0:
    st.success("This dataset has **no missing values** -- enjoy it while it lasts, because in the real world this almost never happens!")
else:
    st.warning(f"Total missing cells: {total_missing:,}")

# Visualize missing by column
fig = px.bar(
    missing,
    x="Column",
    y="Missing %",
    title="Missing Values by Column",
    labels={"Missing %": "Percent Missing"},
)
apply_common_layout(fig, height=350)
st.plotly_chart(fig, use_container_width=True)

code_example(
    """# Count missing values
df.isnull().sum()

# Percentage missing
df.isnull().mean() * 100

# Drop rows with any missing value
df_clean = df.dropna()

# Fill missing with column median
df["temperature_c"].fillna(df["temperature_c"].median(), inplace=True)
"""
)

# ── Interactive column explorer ──────────────────────────────────────────────
st.header("1.7  Interactive Column Explorer")
st.markdown(
    "Pick any column and poke around. You'll see its data type, how many non-null "
    "values it has, how many unique values, and a quick histogram or bar chart. "
    "This is the kind of casual exploration that builds intuition before you start "
    "any formal analysis."
)

explore_col = st.selectbox(
    "Column", fdf.columns.tolist(), key="explore_col"
)

col_a, col_b = st.columns(2)
with col_a:
    st.write(f"**Dtype:** `{fdf[explore_col].dtype}`")
    st.write(f"**Non-null count:** {fdf[explore_col].notna().sum():,}")
    st.write(f"**Unique values:** {fdf[explore_col].nunique():,}")

with col_b:
    if pd.api.types.is_numeric_dtype(fdf[explore_col]):
        fig_hist = px.histogram(
            fdf, x=explore_col, nbins=50,
            title=f"Distribution of {FEATURE_LABELS.get(explore_col, explore_col)}",
        )
        apply_common_layout(fig_hist, height=300)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        vc = fdf[explore_col].value_counts().head(15).reset_index()
        vc.columns = [explore_col, "count"]
        fig_bar = px.bar(vc, x=explore_col, y="count", title=f"Top values: {explore_col}")
        apply_common_layout(fig_bar, height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "What does `df.shape` return?",
    [
        "A list of column names",
        "A tuple of (rows, columns)",
        "A DataFrame of summary statistics",
        "The data type of each column",
    ],
    correct_idx=1,
    explanation="df.shape gives you a tuple: (number_of_rows, number_of_columns). "
                "It's the first thing I check with any new dataset -- just to know how big the thing is.",
    key="ch1_quiz1",
)

quiz(
    "Which method gives you count, mean, std, min, quartiles, and max in one call?",
    [
        "df.info()",
        "df.head()",
        "df.describe()",
        "df.dtypes",
    ],
    correct_idx=2,
    explanation="describe() is your one-stop shop for common summary statistics on numeric columns. "
                "It won't tell you everything, but it'll tell you enough to know if something is obviously wrong.",
    key="ch1_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "`df.head()` and `df.tail()` let you peek at the first/last rows -- always start here.",
    "`df.shape` returns (rows, columns); `df.dtypes` reveals the type of each column, which matters more than you'd think.",
    "`df.describe()` gives you count, mean, std, min, quartiles, and max -- a full health check in one line.",
    "Always check for missing values with `df.isnull().sum()` before doing anything else. Future you will thank present you.",
    "`df.info()` gives a concise overview of the entire DataFrame, including memory usage.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Welcome",
    prev_page="00_Welcome.py",
    next_label="Ch 2: Descriptive Statistics",
    next_page="02_Descriptive_Statistics.py",
)
