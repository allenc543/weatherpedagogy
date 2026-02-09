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
    "We have spent the last several chapters building increasingly powerful ensembles "
    "of decision trees. Now we take a sharp turn into a completely different family of "
    "models: **neural networks**. And I want to start by demystifying them, because the "
    "name alone -- 'neural network' -- conjures images of brain-like complexity that is "
    "mostly undeserved at this level."
)
st.markdown(
    "**The task**: We are still working with our weather data. Given a humidity reading, "
    "can we predict the temperature? One number in, one number out. The simplest possible "
    "prediction task. I am going to show you that a single 'neuron' is literally just "
    "linear regression -- the same thing you already know -- and then explain what you "
    "need to add to make it more powerful."
)
st.markdown(
    "By the end of this chapter, you will understand three things: (1) what a neuron "
    "actually computes, (2) why non-linear activation functions are the key ingredient, "
    "and (3) how gradient descent trains the whole thing. And we will test whether a "
    "simple neural network actually beats linear regression on our weather data."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 50.1 The Single Neuron ──────────────────────────────────────────────────
st.header("50.1  A Single Neuron = Linear Regression (Seriously)")

st.markdown(
    "Let me strip away all the mystique. A single artificial neuron does this:"
)
st.markdown(
    "1. Takes inputs (e.g., humidity = 72%)\n"
    "2. Multiplies each input by a weight (e.g., weight = -0.15)\n"
    "3. Adds a bias term (e.g., bias = 35.2)\n"
    "4. Passes the result through an activation function\n\n"
    "With a **linear activation** (which just means 'do nothing'), step 4 is a no-op. "
    "So the output is: -0.15 * 72 + 35.2 = 24.4 degrees C. That is linear regression. "
    "One weight, one bias, one multiply-and-add. The same thing you learned in a "
    "statistics class, just rebranded."
)

concept_box(
    "The Artificial Neuron in Weather Terms",
    "Our neuron is predicting temperature from humidity. The weight tells it the "
    "relationship: 'for each 1% increase in humidity, temperature changes by w degrees.' "
    "A negative weight (-0.15) means 'higher humidity tends to go with lower temperature' "
    "(which makes sense -- cloudy, humid days are often cooler than dry, sunny ones). "
    "The bias is the baseline: 'if humidity were 0%, the predicted temperature would be "
    "35.2 C.'<br><br>"
    "With a linear activation, this is <b>exactly</b> the equation y = wx + b. "
    "A single neuron with a linear activation IS linear regression. Same math, "
    "different vocabulary."
)

formula_box(
    "Single Neuron",
    r"y = \sigma\!\left(\sum_{i=1}^{n} w_i x_i + b\right) = \sigma(\mathbf{w}^T\mathbf{x} + b)",
    "sigma = activation function, w = weights, b = bias, x = inputs. "
    "If sigma is the identity function (linear activation), this collapses to "
    "y = w*x + b, which is linear regression."
)

st.subheader("Interactive: Single Neuron on Temperature vs Humidity")

st.markdown(
    "Select a city below. We will fit both a linear regression and a single neuron "
    "(with linear activation) to that city's humidity-vs-temperature data. "
    "Spoiler: they produce the same line."
)

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
    st.markdown(
        f"This means: for each 1% increase in humidity in {city_neuron}, "
        f"the predicted temperature changes by {lr.coef_[0]:.2f} degrees C."
    )
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
    f"The red solid line (linear regression) and the dark dashed line (single neuron) "
    f"are the same line. They produce identical predictions. For {city_neuron}, the "
    f"relationship between humidity and temperature is captured by one weight and one "
    f"bias. But wait -- you might ask -- if they are the same thing, why do neural "
    f"networks exist? Because the magic happens when you start <b>stacking</b> "
    f"neurons in layers and swap in <b>non-linear activations</b>. That is when a "
    f"neural network can learn curves, kinks, and complex patterns that a single "
    f"straight line cannot. That is the topic of the next two sections."
)

# ── 50.2 Activation Functions ────────────────────────────────────────────────
st.header("50.2  Activation Function Visualizer")

st.markdown(
    "So a single neuron with a linear activation is just linear regression. Why not "
    "stack 10 linear neurons? Would that be more powerful?"
)
st.markdown(
    "No. And this is the key insight. A linear function of a linear function is still "
    "linear. If Layer 1 computes y = 3x + 2, and Layer 2 computes z = 5y - 1, then "
    "z = 5(3x + 2) - 1 = 15x + 9. You have stacked two layers but the result is still "
    "a single straight line. You could stack a hundred linear layers and the result "
    "would collapse into one linear equation. All that complexity, and you have "
    "accomplished exactly nothing."
)

concept_box(
    "Why Non-Linear Activations Are Everything",
    "Non-linear activations break the 'stacking linear layers is useless' problem. "
    "They introduce curves and kinks between layers, so the composition of many layers "
    "can represent arbitrarily complex functions.<br><br>"
    f"For our weather data, the relationship between humidity and temperature is not "
    f"perfectly linear. In {city_neuron}, at low humidity (< 30%), temperatures might be "
    f"quite high (clear, sunny days). At moderate humidity (40-70%), there is a wide "
    f"spread. At very high humidity (> 90%), temperatures cluster in a narrow range "
    f"(fog, rain). A straight line cannot capture these different regimes. But a neural "
    f"network with non-linear activations can learn separate behaviors for each humidity "
    f"range -- essentially learning a different slope for each region of the input space."
)

st.subheader("Interactive: Compare Activation Functions")
st.markdown(
    "Below you can visualize different activation functions and their derivatives. "
    "Each activation transforms the neuron's output in a different way. The derivative "
    "matters because gradient descent uses it to determine how to update the weights "
    "(more on that in Section 50.3)."
)

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

st.markdown(
    "**What to look for in the plots:**\n\n"
    "- **ReLU** (dark blue): Dead simple -- output equals input when positive, zero when "
    "negative. The derivative is either 0 or 1. This is the default choice for hidden "
    "layers in modern networks because it is fast and avoids the 'vanishing gradient' "
    "problem.\n\n"
    "- **Sigmoid** (red): Squishes everything to the range (0, 1). The derivative gets "
    "tiny for large positive or negative inputs (the 'vanishing gradient' problem), which "
    "makes learning slow for neurons with extreme outputs.\n\n"
    "- **Tanh** (green): Like sigmoid but outputs range from -1 to 1. Same vanishing "
    "gradient issue at the extremes."
)

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

st.markdown(
    "We have a neuron with weights and biases. We have an activation function. But "
    "how do we *find* the right weights? How does the network learn that the weight "
    "connecting humidity to temperature should be -0.15 and not +0.30 or -0.02?"
)

concept_box(
    "Gradient Descent: Finding the Right Weights",
    "Imagine you are standing on a hilly landscape in the dark. Your altitude is the "
    "'loss' -- how wrong the model's predictions are. Your goal is to reach the lowest "
    "valley (minimum loss = best predictions). You cannot see the landscape, but you "
    "can feel which direction is downhill under your feet.<br><br>"
    "So you take a step downhill. Feel the slope again. Take another step. Repeat. "
    "That is gradient descent. The network computes the gradient (slope) of the loss "
    "with respect to each weight: 'if I increase this weight by a tiny amount, does "
    "the prediction error go up or down?' Then it adjusts each weight in the direction "
    "that reduces the error.<br><br>"
    "For our weather model: if increasing the humidity weight makes temperature "
    "predictions worse, the gradient tells the network to decrease that weight. If "
    "increasing the bias makes predictions better, increase the bias. Each step nudges "
    "all the weights toward values that produce better predictions."
)

formula_box(
    "Weight Update Rule",
    r"w_{t+1} = w_t - \eta \cdot \frac{\partial \mathcal{L}}{\partial w_t}",
    "eta = learning rate (step size), L = loss function (e.g., mean squared error of "
    "temperature predictions), dL/dw = the gradient telling you which direction is "
    "downhill. This single equation is the entire training algorithm. Everything else "
    "-- backpropagation, Adam optimizer, batch normalization -- is an optimization "
    "of this core idea."
)

st.subheader("Interactive: Learning Rate and Gradient Descent Convergence")

st.markdown(
    "Below is a simple loss landscape: a parabola with its minimum at w = 3.0. "
    "Gradient descent starts at w = 10.0 (far from the optimum) and tries to walk "
    "downhill. The **learning rate** controls how big each step is."
)

lr_gd = st.select_slider(
    "Learning Rate",
    options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
    value=0.1,
    key="gd_lr",
)
n_steps = st.slider("Number of steps", 10, 200, 50, key="gd_steps")

st.markdown(
    f"**Learning rate = {lr_gd}**: Each step moves the weight by {lr_gd} times the "
    f"gradient. Try different values and watch what happens -- too small and it barely "
    f"moves; too large and it overshoots the valley and bounces around."
)

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
        "the weight is bouncing back and forth across the valley, overshooting "
        "every time, like a ball that rolls so fast it flies past the bottom and "
        "ends up higher than where it started. In a real neural network training "
        "on weather data, this would mean the predicted temperatures get worse "
        "with every training step instead of better. The fix: lower the learning rate."
    )
elif lr_gd <= 0.005:
    st.info(
        f"With learning rate = {lr_gd}, each step is tiny. After {n_steps} steps, "
        f"the weight has only moved from 10.0 to {trajectory[-1]:.3f} -- still far "
        f"from the optimal 3.0. You would need hundreds more steps to converge. In "
        f"a real neural network, this means training takes a very long time. The model "
        f"would eventually learn the right humidity-temperature relationship, but you "
        f"might run out of patience first."
    )
elif abs(trajectory[-1] - 3.0) < 0.1:
    st.success(f"Converged to w = {trajectory[-1]:.3f} (near optimal w = 3.0). The learning rate is well-chosen for this landscape.")

# Compare learning rates
st.subheader("Convergence at Different Learning Rates")
st.markdown(
    "Here are all the learning rates side by side. This is the single most important "
    "plot for building intuition about neural network training."
)

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
    "LR=0.001 (purple) barely moves -- after 100 steps it is still far from the "
    "minimum. LR=0.1 (green) converges smoothly in about 30 steps. LR=0.5 (dark blue) "
    "oscillates but eventually settles. LR=1.0 (red) bounces wildly and may diverge "
    "entirely. In real weather model training, LR=0.01 is a common starting point for "
    "neural networks -- conservative enough to converge, fast enough to finish in a "
    "reasonable time. Modern optimizers like Adam adaptively tune the learning rate, "
    "which is why you rarely have to hand-tune it anymore."
)

# ── 50.4 A Simple Neural Network on Weather Data ────────────────────────────
st.header("50.4  Putting It Together: Simple NN on Weather Data")

st.markdown(
    "Enough theory. Let us train actual neural networks on our weather data and "
    "answer the question: does all this machinery actually buy us anything over "
    "plain linear regression for predicting temperature?"
)
st.markdown(
    "**The setup**: Predict temperature from 5 features -- humidity, wind speed, "
    "surface pressure, hour of day, and month. Linear regression can only learn "
    "relationships like 'each 1% humidity increase lowers temperature by 0.15 C.' "
    "A neural network can learn non-linear relationships like 'high humidity lowers "
    "temperature during the day but has less effect at night' or 'the pressure-"
    "temperature relationship is different in summer vs winter.'"
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

st.markdown(
    f"Linear regression achieves RMSE = {lr_rmse:.2f} C. The single-layer neural "
    f"network with 10 neurons achieves RMSE = {nn1_rmse:.2f} C. The two-layer network "
    f"with 32+16 neurons achieves RMSE = {nn2_rmse:.2f} C."
)

if nn2_rmse < lr_rmse:
    st.markdown(
        f"The neural network improves by about {lr_rmse - nn2_rmse:.2f} degrees. "
        f"That is because the relationship between our features and temperature is not "
        f"purely linear. For example, humidity's effect on temperature depends on the "
        f"time of day (hour), and the neural network can capture that interaction. "
        f"Linear regression cannot -- it assumes each feature has a fixed, independent "
        f"effect. But notice the improvement is modest. On tabular weather data with "
        f"only 5 features, you do not need a massive network. The non-linear patterns "
        f"are there, but they are subtle."
    )
else:
    st.markdown(
        "Interestingly, the neural network does not clearly beat linear regression here. "
        "This can happen on tabular data when the relationships are mostly linear and "
        "the dataset is not large enough for the network to learn the subtle non-linear "
        "patterns. Tree-based models (random forest, gradient boosting) often do better "
        "than neural networks on this kind of structured tabular data."
    )

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
    "A single neuron with a linear activation function predicting temperature from humidity is equivalent to:",
    [
        "A decision tree with one split",
        "Linear regression (y = w * humidity + b)",
        "K-nearest neighbors with k=1",
        "A random forest with one tree",
    ],
    correct_idx=1,
    explanation=(
        "A single neuron computes output = activation(w * humidity + b). With a linear "
        "activation (which does nothing), this is just output = w * humidity + b -- "
        "literally the equation for linear regression. The neuron learns a weight (slope) "
        "and bias (intercept), exactly like fitting a line to the humidity-temperature "
        "scatter plot. The power of neural networks comes from stacking many neurons with "
        "non-linear activations, not from a single neuron."
    ),
    key="ch50_quiz1",
)

quiz(
    "Why are non-linear activation functions necessary in neural networks?",
    [
        "They make the network faster to train by simplifying the gradient computation",
        "They reduce the number of parameters the network needs",
        "Without them, stacking layers is equivalent to a single linear transformation -- depth is useless",
        "They prevent overfitting by constraining the output range",
    ],
    correct_idx=2,
    explanation=(
        "This is the most important insight about neural networks: a linear function of "
        "a linear function is still linear. Layer 1: y = 3*humidity + 2. Layer 2: "
        "z = 5*y - 1 = 15*humidity + 9. You stacked two layers but the result is still "
        "a straight line. You could stack a thousand linear layers and it would all "
        "collapse into a single linear equation. Non-linear activations (ReLU, sigmoid, "
        "tanh) break this degeneracy. They introduce curves and kinks that allow the "
        "network to learn that humidity affects temperature differently at different "
        "times of day or in different humidity ranges."
    ),
    key="ch50_quiz2",
)

quiz(
    "You set the learning rate to 2.0 and the loss increases with every training step. What is happening?",
    [
        "The model needs more layers to learn the weather patterns",
        "The gradient computation is incorrect",
        "The learning rate is too large -- gradient descent is overshooting the minimum and diverging",
        "The data needs to be normalized before training",
    ],
    correct_idx=2,
    explanation=(
        "A learning rate of 2.0 is like taking enormous leaps downhill in the dark. "
        "Each step overshoots the valley floor and lands on the opposite hillside, even "
        "higher than before. The next step overshoots again, even further. The loss "
        "increases with every step because the weight updates are too large -- the model "
        "is bouncing further and further from the optimal weights for predicting temperature. "
        "The fix: lower the learning rate to 0.01 or 0.001. On our weather data, a "
        "learning rate of 0.01 with 500 iterations typically converges nicely."
    ),
    key="ch50_quiz3",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "A **single neuron** with a linear activation is exactly linear regression: it learns one weight per feature (e.g., 'each 1% humidity increase changes temperature by -0.15 C') and a bias.",
    "**Non-linear activations** (ReLU, sigmoid, tanh) are the ingredient that makes depth useful. Without them, stacking layers is equivalent to a single linear transformation -- all the architecture accomplishes nothing.",
    "**ReLU** is the default activation for hidden layers: simple (zero if negative, identity if positive), fast, and avoids the vanishing gradient problem that plagues sigmoid/tanh.",
    "**Gradient descent** finds the right weights by repeatedly computing 'which direction reduces the prediction error?' and taking a step in that direction. The learning rate controls step size.",
    "The **learning rate** is the most consequential hyperparameter: too small means training takes forever (the humidity-temperature weights barely change per step); too large means the weights oscillate and the model gets worse, not better.",
    "On tabular weather data, neural networks can beat linear regression by capturing non-linear feature interactions, but the improvement is often modest. Tree-based models frequently outperform neural networks on structured data.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 49: Stacking",
    prev_page="49_Stacking.py",
    next_label="Ch 51: Feedforward Networks",
    next_page="51_Feedforward_Networks.py",
)
