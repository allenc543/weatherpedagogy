"""Machine learning model training and evaluation wrappers."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, roc_curve, precision_recall_fscore_support,
)


def prepare_classification_data(df, features, target="city", test_size=0.2, scale=True, seed=42):
    """Prepare data for classification: encode target, split, optionally scale."""
    le = LabelEncoder()
    X = df[features].dropna()
    y = le.fit_transform(df.loc[X.index, target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

    return X_train, X_test, y_train, y_test, le, scaler


def prepare_regression_data(df, features, target, test_size=0.2, scale=False, seed=42):
    """Prepare data for regression: split and optionally scale."""
    clean = df[features + [target]].dropna()
    X = clean[features]
    y = clean[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

    return X_train, X_test, y_train, y_test, scaler


def classification_metrics(y_true, y_pred, labels=None):
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


def regression_metrics(y_true, y_pred):
    """Compute regression metrics."""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def plot_confusion_matrix(cm, labels):
    """Return a Plotly heatmap of a confusion matrix."""
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues", text=cm, texttemplate="%{text}",
    ))
    fig.update_layout(
        xaxis_title="Predicted", yaxis_title="Actual",
        title="Confusion Matrix", height=500,
        template="plotly_white",
    )
    return fig


@st.cache_resource
def train_model(model_class, X_train, y_train, **kwargs):
    """Train and cache a model."""
    model = model_class(**kwargs)
    model.fit(X_train, y_train)
    return model
