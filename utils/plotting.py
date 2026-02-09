"""Shared Plotly plotting helpers."""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.constants import CITY_COLORS, FEATURE_LABELS


def apply_common_layout(fig, title=None, height=500):
    """Apply common layout settings to a Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=title,
        title_x=0.5,
        margin=dict(t=60, b=40, l=60, r=40),
    )
    return fig


def color_map():
    """Return the city color discrete map for Plotly."""
    return CITY_COLORS


def line_chart(df, x, y, color="city", title=None, labels=None, height=500):
    """Create a line chart with city colors."""
    lab = {**(labels or {})}
    for k, v in FEATURE_LABELS.items():
        lab.setdefault(k, v)
    fig = px.line(df, x=x, y=y, color=color, color_discrete_map=CITY_COLORS,
                  labels=lab, title=title)
    return apply_common_layout(fig, title, height)


def scatter_chart(df, x, y, color="city", title=None, labels=None, height=500, opacity=0.3):
    """Create a scatter plot with city colors."""
    lab = {**(labels or {})}
    for k, v in FEATURE_LABELS.items():
        lab.setdefault(k, v)
    fig = px.scatter(df, x=x, y=y, color=color, color_discrete_map=CITY_COLORS,
                     labels=lab, title=title, opacity=opacity)
    return apply_common_layout(fig, title, height)


def histogram_chart(df, x, color="city", title=None, nbins=50, labels=None, height=500, marginal=None):
    """Create a histogram with city colors."""
    lab = {**(labels or {})}
    for k, v in FEATURE_LABELS.items():
        lab.setdefault(k, v)
    fig = px.histogram(df, x=x, color=color, color_discrete_map=CITY_COLORS,
                       nbins=nbins, labels=lab, title=title, barmode="overlay",
                       opacity=0.7, marginal=marginal)
    return apply_common_layout(fig, title, height)


def box_chart(df, x, y, color=None, title=None, labels=None, height=500):
    """Create a box plot."""
    lab = {**(labels or {})}
    for k, v in FEATURE_LABELS.items():
        lab.setdefault(k, v)
    fig = px.box(df, x=x, y=y, color=color or x,
                 color_discrete_map=CITY_COLORS, labels=lab, title=title)
    return apply_common_layout(fig, title, height)


def heatmap_chart(data, x_label="", y_label="", title=None, height=500, color_scale="RdYlBu_r"):
    """Create a heatmap from a 2D array or DataFrame."""
    fig = go.Figure(data=go.Heatmap(
        z=data.values if hasattr(data, 'values') else data,
        x=data.columns.tolist() if hasattr(data, 'columns') else None,
        y=data.index.tolist() if hasattr(data, 'index') else None,
        colorscale=color_scale,
    ))
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
    return apply_common_layout(fig, title, height)


def multi_subplot(rows, cols, subplot_titles=None, shared_xaxes=False, shared_yaxes=False):
    """Create a subplot figure."""
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles,
                        shared_xaxes=shared_xaxes, shared_yaxes=shared_yaxes)
    return fig
