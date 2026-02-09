"""Chapter 50: Neural Network Basics -- Neurons, activations, gradient descent."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(50, "Neural Network Basics", part="XII")
st.markdown(
    "There is a running joke in machine learning that neural networks are "
    "\"just matrix multiplications.\" This is true in roughly the same way that "
    "the Mona Lisa is \"just pigment on wood.\" Technically correct, but it "
    "leaves out the interesting part. This chapter starts from the simplest "
    "building block -- a **single neuron** -- and works up to **activation "
    "functions** and **gradient descent**, the algorithm that makes the whole "
    "enterprise possible."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 50.1 The Single Neuron ──────────────────────────────────────────────────
st.header("50.1  A Single Neuron = Linear Regression")

concept_box(
    "The Artificial Neuron",
    "A single neuron is just a linear regression wearing a trench coat. "
    "It multiplies each input by a weight, adds them up, tosses in a bias term, "
    "and passes the result through an activation function:<br><br>"
    "<code>output = activation(w1*x1 + w2*x2 + ... + b)</code><br><br>"
    "Strip away the mystique and that is all there is. With a linear (identity) "
    "activation, this is <b>exactly</b> linear regression. The same thing you "
    "learned in stats class, just rebranded with fancier vocabulary."
)

formula_box(
    "Single Neuron",
    r"y = \sigma\!\left(\sum_{i=1}^{n} w_i x_i + b\right) = \sigma(\mathbf{w}^T\mathbf{x} + b)",
    "sigma = activation function, w = weights, b = bias, x = inputs. "
    "If sigma is the identity function, congratulations, you have reinvented linear regression."
)

st.subheader("Interactive: Single Neuron on Temperature vs Humidity")

city_neuron = st.selectbox("City", CITY_LIST, key="neuron_city")
city_data = fdf[fdf["city"] == city_neuron].dropna(subset=["temperature_c", "relative_humidity_pct"])

# Subsample for visualization
sample_n = min(2000, len(city_data))
city_sample = city_data.sample(sample_n, random_state=42)

X_neuron = city_sample[["relative_humidity_pct"]].values
y_neuron = city_sample["temperature_c"].values

# Linear regression (= single neuron with linear activation)
lr = LinearRegression()
lr.fit(X_neuron, y_neuron)
lr_pred = lr.predict(X_neuron)

# Single neuron MLP (1 hidden layer with 1 neuron, identity activation)
scaler_n = StandardScaler()
X_scaled = scaler_n.fit_transform(X_neuron)

mlp_single = MLPRegressor(
    hidden_layer_sizes=(), max_iter=500, random_state=42,
    activation="identity",
)
mlp_single.fit(X_scaled, y_neuron)
mlp_pred = mlp_single.predict(X_scaled)

col1, col2 = st.columns(2)
with col1:
    st.metric("Linear Regression R-squared", f"{r2_score(y_neuron, lr_pred):.4f}")
    st.markdown(f"**Weight (w):** {lr.coef_[0]:.4f}")
    st.markdown(f"**Bias (b):** {lr.intercept_:.4f}")
with col2:
    st.metric("Single Neuron R-squared", f"{r2_score(y_neuron, mlp_pred):.4f}")

fig_neuron = go.Figure()
fig_neuron.add_trace(go.Scatter(
    x=city_sample["relative_humidity_pct"], y=city_sample["temperature_c"],
    mode="markers", marker=dict(size=3, color="#2A9D8F", opacity=0.3),
    name="Data",
))
x_range = np.linspace(X_neuron.min(), X_neuron.max(), 100).reshape(-1, 1)
fig_neuron.add_trace(go.Scatter(
    x=x_range.flatten(), y=lr.predict(x_range),
    mode="lines", line=dict(color="#E63946", width=2),
    name="Linear Regression",
))
fig_neuron.add_trace(go.Scatter(
    x=x_range.flatten(), y=mlp_single.predict(scaler_n.transform(x_range)),
    mode="lines", line=dict(color="#264653", width=2, dash="dash"),
    name="Single Neuron (linear activation)",
))
apply_common_layout(fig_neuron, title=f"Single Neuron = Linear Regression -- {city_neuron}", height=450)
fig_neuron.update_layout(xaxis_title="Relative Humidity (%)", yaxis_title="Temperature (C)")
st.plotly_chart(fig_neuron, use_container_width=True)

insight_box(
    "Look at those two lines. They are the same line. A single neuron with a "
    "linear activation produces an **identical** fit to ordinary linear regression. "
    "You might ask: if they are the same thing, why bother with the neuron framing? "
    "Because the magic happens when you start **stacking** these neurons in layers "
    "and swap in **non-linear activations**. That is when things get interesting."
)

# ── 50.2 Activation Functions ────────────────────────────────────────────────
st.header("50.2  Activation Function Visualizer")

concept_box(
    "Why Activation Functions?",
    "Here is a fact that sounds obvious once you hear it but has profound "
    "consequences: a linear function of linear functions is still linear. "
    "Stack a hundred linear layers and you have accomplished exactly nothing "
    "that a single layer could not do. <b>Non-linear activations</b> are the "
    "ingredient that breaks this degeneracy. They are what allow deep networks "
    "to learn curved, kinked, and generally interesting decision boundaries "
    "instead of just hyperplanes."
)

st.subheader("Interactive: Compare Activation Functions")
activations_to_show = st.multiselect(
    "Select activations to visualize",
    ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "ELU", "Swish"],
    default=["Sigmoid", "ReLU", "Tanh"],
    key="act_select",
)

x_act = np.linspace(-5, 5, 300)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def swish(x):
    return x * sigmoid(x)

activation_funcs = {
    "Sigmoid": (sigmoid, "#E63946"),
    "Tanh": (tanh, "#2A9D8F"),
    "ReLU": (relu, "#264653"),
    "Leaky ReLU": (leaky_relu, "#FB8500"),
    "ELU": (elu, "#7209B7"),
    "Swish": (swish, "#F4A261"),
}

fig_act = make_subplots(rows=1, cols=2, subplot_titles=["Activation Functions", "Derivatives"])

for name in activations_to_show:
    func, color = activation_funcs[name]
    y_act = func(x_act)
    # Derivative (numerical)
    dy = np.gradient(y_act, x_act)

    fig_act.add_trace(go.Scatter(
        x=x_act, y=y_act, mode="lines", name=name,
        line=dict(color=color, width=2),
    ), row=1, col=1)
    fig_act.add_trace(go.Scatter(
        x=x_act, y=dy, mode="lines", name=f"{name}'",
        line=dict(color=color, width=2, dash="dash"),
        showlegend=False,
    ), row=1, col=2)

fig_act.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
fig_act.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2)
fig_act.add_vline(x=0, line_dash="dot", line_color="gray", row=1, col=1)
fig_act.add_vline(x=0, line_dash="dot", line_color="gray", row=1, col=2)
apply_common_layout(fig_act, title="Activation Functions and Their Derivatives", height=400)
st.plotly_chart(fig_act, use_container_width=True)

# Properties table
props = pd.DataFrame({
    "Activation": ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU"],
    "Range": ["(0, 1)", "(-1, 1)", "[0, inf)", "(-inf, inf)"],
    "Vanishing Gradient?": ["Yes (large |x|)", "Yes (large |x|)", "No (for x>0)", "No"],
    "Dead Neurons?": ["No", "No", "Yes (x<0)", "No"],
    "Common Use": ["Output (binary)", "Hidden layers (older nets)", "Default hidden layer", "Alternative to ReLU"],
})
st.dataframe(props, use_container_width=True, hide_index=True)

# ── 50.3 Gradient Descent ───────────────────────────────────────────────────
st.header("50.3  Gradient Descent Visualizer")

concept_box(
    "Gradient Descent",
    "Imagine you are blindfolded on a hilly landscape and your goal is to reach "
    "the lowest valley. You cannot see, but you can feel which direction is "
    "downhill. So you take a step downhill, feel again, step again. That is "
    "gradient descent. The network computes which direction in weight-space "
    "reduces the loss, then takes a step in that direction. The <b>learning "
    "rate</b> controls how big each step is -- and as we are about to see, "
    "getting this wrong is spectacularly entertaining."
)

formula_box(
    "Weight Update Rule",
    r"w_{t+1} = w_t - \eta \cdot \frac{\partial \mathcal{L}}{\partial w_t}",
    "eta = learning rate (step size), L = loss function. "
    "This is the entire algorithm. Everything else is details."
)

st.subheader("Interactive: Learning Rate and Gradient Descent Convergence")

lr_gd = st.select_slider(
    "Learning Rate",
    options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
    value=0.1,
    key="gd_lr",
)
n_steps = st.slider("Number of steps", 10, 200, 50, key="gd_steps")

# Simple 1D quadratic: f(w) = (w - 3)^2 + 1
# Gradient: f'(w) = 2(w - 3)
def loss_fn(w):
    return (w - 3) ** 2 + 1

def grad_fn(w):
    return 2 * (w - 3)

# Run gradient descent
w = 10.0  # start far from optimum
trajectory = [w]
losses = [loss_fn(w)]

for _ in range(n_steps):
    g = grad_fn(w)
    w = w - lr_gd * g
    trajectory.append(w)
    losses.append(loss_fn(w))

# Plot loss landscape and trajectory
w_range = np.linspace(-5, 15, 200)
loss_range = loss_fn(w_range)

fig_gd = make_subplots(rows=1, cols=2,
                         subplot_titles=["Loss Landscape", "Loss vs Steps"])

fig_gd.add_trace(go.Scatter(
    x=w_range, y=loss_range, mode="lines",
    line=dict(color="#264653"), name="Loss f(w)",
), row=1, col=1)

# Show trajectory
traj_arr = np.array(trajectory)
fig_gd.add_trace(go.Scatter(
    x=traj_arr, y=loss_fn(traj_arr),
    mode="lines+markers", marker=dict(size=5, color="#E63946"),
    line=dict(color="#E63946"), name="GD Path",
), row=1, col=1)
fig_gd.add_trace(go.Scatter(
    x=[traj_arr[0]], y=[loss_fn(traj_arr[0])],
    mode="markers", marker=dict(size=12, color="green", symbol="star"),
    name="Start", showlegend=True,
), row=1, col=1)
fig_gd.add_trace(go.Scatter(
    x=[traj_arr[-1]], y=[loss_fn(traj_arr[-1])],
    mode="markers", marker=dict(size=12, color="red", symbol="diamond"),
    name="End", showlegend=True,
), row=1, col=1)

# Loss over iterations
fig_gd.add_trace(go.Scatter(
    x=list(range(len(losses))), y=losses,
    mode="lines+markers", marker=dict(size=3),
    line=dict(color="#2A9D8F"), name="Loss",
), row=1, col=2)

apply_common_layout(fig_gd, title=f"Gradient Descent (LR={lr_gd})", height=400)
fig_gd.update_xaxes(title_text="Weight (w)", row=1, col=1)
fig_gd.update_yaxes(title_text="Loss", row=1, col=1)
fig_gd.update_xaxes(title_text="Step", row=1, col=2)
fig_gd.update_yaxes(title_text="Loss", row=1, col=2)
st.plotly_chart(fig_gd, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Starting Loss", f"{losses[0]:.3f}")
c2.metric("Final Loss", f"{losses[-1]:.3f}")
c3.metric("Final Weight", f"{trajectory[-1]:.3f} (optimal: 3.0)")

if lr_gd >= 1.0 and losses[-1] > losses[0]:
    warning_box(
        "The learning rate is too high and gradient descent is **diverging** -- "
        "bouncing back and forth like a ball rolling so fast it overshoots the "
        "valley and ends up higher than where it started. This is the most "
        "common failure mode in neural network training."
    )
elif lr_gd <= 0.005:
    st.info(
        "With this tiny learning rate, you are essentially tiptoeing toward the "
        "optimum. Admirable caution, but you might die of old age before converging. "
        "You may need more steps to reach the minimum."
    )
elif abs(trajectory[-1] - 3.0) < 0.1:
    st.success(f"Converged to w = {trajectory[-1]:.3f} (near optimal w = 3.0)")

# Compare learning rates
st.subheader("Convergence at Different Learning Rates")
lr_values = [0.001, 0.01, 0.1, 0.5, 1.0]
lr_colors = ["#7209B7", "#FB8500", "#2A9D8F", "#264653", "#E63946"]

fig_lr_comp = go.Figure()
for lr_val, color in zip(lr_values, lr_colors):
    w = 10.0
    loss_hist = [loss_fn(w)]
    for _ in range(100):
        w = w - lr_val * grad_fn(w)
        loss_hist.append(loss_fn(w))
    fig_lr_comp.add_trace(go.Scatter(
        x=list(range(len(loss_hist))), y=loss_hist,
        mode="lines", name=f"LR={lr_val}",
        line=dict(color=color),
    ))
apply_common_layout(fig_lr_comp, title="Loss Convergence at Different Learning Rates", height=400)
fig_lr_comp.update_layout(xaxis_title="Step", yaxis_title="Loss",
                           yaxis_range=[0, max(50, losses[0] * 1.5)])
st.plotly_chart(fig_lr_comp, use_container_width=True)

insight_box(
    "The learning rate is arguably the single most important hyperparameter in "
    "all of deep learning. Too small and training takes geological time. Too "
    "large and training diverges into numerical chaos. This is why modern "
    "optimizers like Adam and RMSProp exist -- they adaptively tune the learning "
    "rate so you do not have to babysit it yourself. They are not magic, but "
    "they are close enough for government work."
)

# ── 50.4 A Simple Neural Network on Weather Data ────────────────────────────
st.header("50.4  Putting It Together: Simple NN on Weather Data")

st.markdown(
    "Alright, enough theory. Let us train an actual neural network on actual "
    "weather data and see whether all this machinery buys us anything over "
    "plain linear regression. Spoiler: it does, but the margin might surprise you."
)

features_nn = ["relative_humidity_pct", "wind_speed_kmh", "surface_pressure_hpa", "hour", "month"]
nn_data = fdf[features_nn + ["temperature_c"]].dropna()
sample_n = min(8000, len(nn_data))
nn_sample = nn_data.sample(sample_n, random_state=42)

X_nn = nn_sample[features_nn].values
y_nn = nn_sample["temperature_c"].values
X_tr, X_te, y_tr, y_te = train_test_split(X_nn, y_nn, test_size=0.2, random_state=42)

scaler_nn = StandardScaler()
X_tr_s = scaler_nn.fit_transform(X_tr)
X_te_s = scaler_nn.transform(X_te)

# Linear regression baseline
lr_base = LinearRegression()
lr_base.fit(X_tr_s, y_tr)
lr_rmse = np.sqrt(mean_squared_error(y_te, lr_base.predict(X_te_s)))
lr_r2 = r2_score(y_te, lr_base.predict(X_te_s))

# Single-layer NN
nn_1 = MLPRegressor(hidden_layer_sizes=(10,), max_iter=500, random_state=42, learning_rate_init=0.01)
nn_1.fit(X_tr_s, y_tr)
nn1_rmse = np.sqrt(mean_squared_error(y_te, nn_1.predict(X_te_s)))
nn1_r2 = r2_score(y_te, nn_1.predict(X_te_s))

# Two-layer NN
nn_2 = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42, learning_rate_init=0.01)
nn_2.fit(X_tr_s, y_tr)
nn2_rmse = np.sqrt(mean_squared_error(y_te, nn_2.predict(X_te_s)))
nn2_r2 = r2_score(y_te, nn_2.predict(X_te_s))

nn_comp = pd.DataFrame({
    "Model": ["Linear Regression", "NN (1 layer, 10 neurons)", "NN (2 layers, 32+16 neurons)"],
    "RMSE (C)": [lr_rmse, nn1_rmse, nn2_rmse],
    "R-squared": [lr_r2, nn1_r2, nn2_r2],
})
st.dataframe(
    nn_comp.style.format({"RMSE (C)": "{:.3f}", "R-squared": "{:.4f}"})
    .highlight_min(subset=["RMSE (C)"], color="#d4edda")
    .highlight_max(subset=["R-squared"], color="#d4edda"),
    use_container_width=True, hide_index=True,
)

fig_nn = px.bar(nn_comp, x="Model", y="RMSE (C)", color="Model",
                color_discrete_sequence=["#264653", "#2A9D8F", "#E63946"],
                title="RMSE Comparison: Linear Regression vs Neural Networks")
apply_common_layout(fig_nn, height=400)
st.plotly_chart(fig_nn, use_container_width=True)

code_example("""from sklearn.neural_network import MLPRegressor

# Simple neural network
nn = MLPRegressor(
    hidden_layer_sizes=(32, 16),  # 2 hidden layers
    activation='relu',
    learning_rate_init=0.01,
    max_iter=500,
    random_state=42,
)
nn.fit(X_train_scaled, y_train)
predictions = nn.predict(X_test_scaled)
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "A single neuron with a linear activation function is equivalent to:",
    [
        "A decision tree",
        "Linear regression",
        "K-nearest neighbors",
        "Random forest",
    ],
    correct_idx=1,
    explanation=(
        "A single neuron computes w*x + b and applies an activation. With a "
        "linear (identity) activation, this is literally y = w*x + b. "
        "You have traveled a long road to arrive exactly where you started. "
        "The power comes from stacking neurons and adding non-linearity."
    ),
    key="ch50_quiz1",
)

quiz(
    "Why are non-linear activation functions necessary?",
    [
        "They make the network faster to train",
        "They reduce the number of parameters",
        "Without them, stacking layers is equivalent to a single linear transformation",
        "They prevent overfitting",
    ],
    correct_idx=2,
    explanation=(
        "This is the key insight: a linear function of a linear function is "
        "still linear. You could stack a thousand linear layers and the result "
        "would collapse into a single matrix multiplication. Non-linear "
        "activations are the entire reason deep networks can learn anything "
        "that shallow ones cannot."
    ),
    key="ch50_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "A **single neuron** is weighted sum + bias + activation. That is the whole thing.",
    "With a linear activation, a neuron is **exactly** linear regression -- same math, different branding.",
    "**Non-linear activations** (ReLU, sigmoid, tanh) are the secret ingredient that makes depth useful.",
    "**Gradient descent** is just \"feel which way is downhill, take a step, repeat.\"",
    "The **learning rate** is the most consequential hyperparameter: too small = glacial, too large = chaos.",
    "Even a simple MLP can beat linear regression by capturing non-linear relationships -- but do not expect miracles on tabular data.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 49: Stacking",
    prev_page="49_Stacking.py",
    next_label="Ch 51: Feedforward Networks",
    next_page="51_Feedforward_Networks.py",
)
