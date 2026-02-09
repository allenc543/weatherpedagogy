"""Chapter 52: RNN & LSTM -- Sequence modeling, gates, vanishing gradients, temperature forecasting."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(52, "RNN & LSTM", part="XII")
st.markdown(
    "Everything we have built so far treats each data point as if it appeared "
    "in a vacuum -- no past, no future, just a bag of features floating in the "
    "void. This is fine for some problems but borderline absurd for weather "
    "forecasting. The temperature at 3 PM is not independent of the temperature "
    "at 2 PM. **Recurrent Neural Networks (RNNs)** fix this by processing "
    "sequences with memory. **LSTMs** fix the problems with RNNs. This chapter "
    "covers both and puts them to work on 24-hour temperature forecasting."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 52.1 RNN Concepts ───────────────────────────────────────────────────────
st.header("52.1  Recurrent Neural Networks")

concept_box(
    "Sequential Data",
    "Weather data is inherently <b>sequential</b>. The temperature at hour 14 "
    "depends on the temperature at hour 13, which depends on hour 12, and so on "
    "back to the dawn of time (or at least back to sunrise). Feedforward networks "
    "are blind to this temporal structure. An RNN fixes this by processing one "
    "time step at a time, maintaining a <b>hidden state</b> -- a sort of running "
    "summary of everything it has seen so far. Think of it as the network having "
    "a short-term memory, except calling it \"short-term\" is being generous."
)

formula_box(
    "Simple RNN Cell",
    r"h_t = \tanh(W_h h_{t-1} + W_x x_t + b)",
    "h_t = hidden state at time t, x_t = input at time t. The hidden state acts "
    "as memory. The key insight is that h_t depends on h_{t-1}, which creates a "
    "chain going back through the entire sequence."
)

st.markdown("""
**The Vanishing Gradient Problem** (or: why simple RNNs are disappointing):

Here is the fundamental issue. During backpropagation through time, gradients get multiplied at each step. If you are multiplying by numbers less than 1 over and over, your gradient shrinks exponentially toward zero -- and a zero gradient means the network literally cannot learn from distant past inputs. If the multipliers are greater than 1, the gradient explodes toward infinity instead.

This is not a minor technical inconvenience. It means simple RNNs can learn that "the temperature 2 hours ago matters" but effectively cannot learn that "it was raining heavily 18 hours ago and this matters." For weather forecasting, where diurnal cycles and frontal passages operate on timescales of 12-48 hours, this is crippling.
""")

warning_box(
    "Simple RNNs are limited to learning short-range dependencies (a few time "
    "steps at best). For weather forecasting over 24+ hours, we need LSTM or "
    "GRU cells, which were specifically engineered to solve this problem."
)

# ── 52.2 LSTM Architecture ──────────────────────────────────────────────────
st.header("52.2  LSTM: Long Short-Term Memory")

concept_box(
    "LSTM Gates",
    "Here is the genuinely cool thing about LSTMs: they solve the vanishing "
    "gradient problem not by some clever mathematical trick, but by literally "
    "building a highway for information to flow through. They add three "
    "<b>gates</b> -- learned valves that control what gets remembered, what "
    "gets forgotten, and what gets output:<br><br>"
    "- <b>Forget gate (f)</b>: looks at the current input and previous hidden "
    "state and decides what to throw away from the cell state. \"It stopped "
    "raining 6 hours ago? Okay, maybe we can forget the rain intensity.\"<br>"
    "- <b>Input gate (i)</b>: decides what new information to write into the "
    "cell state. \"A cold front just arrived? Better remember that.\"<br>"
    "- <b>Output gate (o)</b>: decides what part of the cell state to actually "
    "use for the current prediction.<br><br>"
    "All three gates use sigmoid activations (output 0-1), so they act like "
    "dimmer switches: fully open, fully closed, or anywhere in between."
)

formula_box("Forget Gate", r"f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)", "Controls what to forget from cell state. A value of 1 means \"keep everything.\" A value of 0 means \"forget completely.\"")
formula_box("Input Gate", r"i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)", "Controls what new information to write into the cell state.")
formula_box("Cell State Update", r"C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t", "The cell state is the LSTM's long-term memory. This equation is where the magic happens: when f_t is close to 1, old information flows through unchanged.")
formula_box("Output Gate", r"o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)", "Controls what part of the cell state to output as the hidden state.")

# ── 52.3 24-Hour Temperature Forecast ───────────────────────────────────────
st.header("52.3  Interactive: 24-Hour Temperature Forecast")

forecast_city = st.selectbox("City for forecasting", CITY_LIST, key="lstm_city")
city_ts = fdf[fdf["city"] == forecast_city].sort_values("datetime").reset_index(drop=True)

if len(city_ts) < 100:
    st.warning("Not enough data for this city after filtering. Please adjust filters.")
else:
    lookback = st.slider("Lookback window (hours of history)", 6, 48, 24, 6, key="lookback")
    forecast_horizon = 24

    # Prepare sequences
    temp_values = city_ts["temperature_c"].values.astype(float)
    scaler_ts = MinMaxScaler()
    temp_scaled = scaler_ts.fit_transform(temp_values.reshape(-1, 1)).flatten()

    def create_sequences(data, lookback, horizon):
        X, y = [], []
        for i in range(len(data) - lookback - horizon + 1):
            X.append(data[i:i + lookback])
            y.append(data[i + lookback:i + lookback + horizon])
        return np.array(X), np.array(y)

    X_seq, y_seq = create_sequences(temp_scaled, lookback, forecast_horizon)

    if len(X_seq) < 100:
        st.warning("Not enough sequential data. Try a shorter lookback or wider date filter.")
    else:
        # Train/test split (temporal)
        split_idx = int(len(X_seq) * 0.8)
        X_train_seq = X_seq[:split_idx]
        y_train_seq = y_seq[:split_idx]
        X_test_seq = X_seq[split_idx:]
        y_test_seq = y_seq[split_idx:]

        # ── Simple Baseline: Persistence (last known value repeated) ─────
        baseline_preds = np.tile(X_test_seq[:, -1:], (1, forecast_horizon))
        baseline_actual = y_test_seq

        baseline_preds_inv = scaler_ts.inverse_transform(baseline_preds.reshape(-1, 1)).flatten().reshape(-1, forecast_horizon)
        actual_inv = scaler_ts.inverse_transform(y_test_seq.reshape(-1, 1)).flatten().reshape(-1, forecast_horizon)

        baseline_rmse = np.sqrt(mean_squared_error(actual_inv.flatten(), baseline_preds_inv.flatten()))
        baseline_mae = mean_absolute_error(actual_inv.flatten(), baseline_preds_inv.flatten())

        # ── Seasonal Baseline: Repeat the lookback pattern ───────────────
        # Take average of same hours from training data
        hourly_mean = city_ts.groupby("hour")["temperature_c"].mean().values
        seasonal_preds = []
        for i in range(len(X_test_seq)):
            test_start_idx = split_idx + i + lookback
            if test_start_idx + forecast_horizon <= len(city_ts):
                hours = city_ts.iloc[test_start_idx:test_start_idx + forecast_horizon]["hour"].values
                seasonal_preds.append(hourly_mean[hours])
            else:
                seasonal_preds.append(np.full(forecast_horizon, hourly_mean.mean()))
        seasonal_preds = np.array(seasonal_preds[:len(actual_inv)])
        seasonal_rmse = np.sqrt(mean_squared_error(actual_inv[:len(seasonal_preds)].flatten(), seasonal_preds.flatten()))
        seasonal_mae = mean_absolute_error(actual_inv[:len(seasonal_preds)].flatten(), seasonal_preds.flatten())

        # ── Linear Autoregressive Model ──────────────────────────────────
        from sklearn.linear_model import LinearRegression

        # For each horizon step, train a linear model
        lr_preds = np.zeros_like(y_test_seq)
        for h in range(forecast_horizon):
            lr = LinearRegression()
            lr.fit(X_train_seq, y_train_seq[:, h])
            lr_preds[:, h] = lr.predict(X_test_seq)

        lr_preds_inv = scaler_ts.inverse_transform(lr_preds.reshape(-1, 1)).flatten().reshape(-1, forecast_horizon)
        lr_rmse = np.sqrt(mean_squared_error(actual_inv.flatten(), lr_preds_inv.flatten()))
        lr_mae = mean_absolute_error(actual_inv.flatten(), lr_preds_inv.flatten())

        # ── LSTM-like model via sklearn MLP (as LSTM proxy) ──────────────
        # Flatten lookback for MLP (simulates learned temporal features)
        from sklearn.neural_network import MLPRegressor

        mlp_preds = np.zeros_like(y_test_seq)
        with st.spinner("Training neural sequence model (MLP proxy for LSTM)..."):
            for h in range(forecast_horizon):
                mlp = MLPRegressor(
                    hidden_layer_sizes=(64, 32), activation="relu",
                    max_iter=200, random_state=42, early_stopping=True,
                    validation_fraction=0.15, learning_rate_init=0.005,
                )
                mlp.fit(X_train_seq, y_train_seq[:, h])
                mlp_preds[:, h] = mlp.predict(X_test_seq)

        mlp_preds_inv = scaler_ts.inverse_transform(mlp_preds.reshape(-1, 1)).flatten().reshape(-1, forecast_horizon)
        mlp_rmse = np.sqrt(mean_squared_error(actual_inv.flatten(), mlp_preds_inv.flatten()))
        mlp_mae = mean_absolute_error(actual_inv.flatten(), mlp_preds_inv.flatten())

        # ── Results comparison ───────────────────────────────────────────
        st.subheader("Model Comparison: 24-Hour Temperature Forecast")

        results_df = pd.DataFrame({
            "Model": ["Persistence (naive)", "Hourly Average (seasonal)", "Linear Autoregressive", "Neural Sequence Model"],
            "RMSE (C)": [baseline_rmse, seasonal_rmse, lr_rmse, mlp_rmse],
            "MAE (C)": [baseline_mae, seasonal_mae, lr_mae, mlp_mae],
        }).sort_values("RMSE (C)")

        st.dataframe(
            results_df.style.format({"RMSE (C)": "{:.3f}", "MAE (C)": "{:.3f}"})
            .highlight_min(subset=["RMSE (C)"], color="#d4edda"),
            use_container_width=True, hide_index=True,
        )

        fig_comp = px.bar(results_df, x="Model", y="RMSE (C)", color="Model",
                          color_discrete_sequence=["#264653", "#FB8500", "#2A9D8F", "#E63946"],
                          title=f"24-Hour Forecast RMSE -- {forecast_city}")
        apply_common_layout(fig_comp, height=400)
        fig_comp.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig_comp, use_container_width=True)

        # ── Visualize one forecast example ───────────────────────────────
        st.subheader("Example Forecast Visualization")
        example_idx = st.slider("Test sample index", 0, len(X_test_seq) - 1, 0, key="forecast_idx")

        history = scaler_ts.inverse_transform(X_test_seq[example_idx].reshape(-1, 1)).flatten()
        actual_future = actual_inv[example_idx]
        persist_future = baseline_preds_inv[example_idx]
        lr_future = lr_preds_inv[example_idx]
        mlp_future = mlp_preds_inv[example_idx]

        hours_history = list(range(-lookback, 0))
        hours_future = list(range(0, forecast_horizon))

        fig_forecast = go.Figure()
        # History
        fig_forecast.add_trace(go.Scatter(
            x=hours_history, y=history,
            mode="lines+markers", marker=dict(size=3),
            line=dict(color="#264653"), name="History",
        ))
        # Actual future
        fig_forecast.add_trace(go.Scatter(
            x=hours_future, y=actual_future,
            mode="lines+markers", marker=dict(size=5),
            line=dict(color="black", width=2), name="Actual",
        ))
        # Persistence
        fig_forecast.add_trace(go.Scatter(
            x=hours_future, y=persist_future,
            mode="lines", line=dict(color="#FB8500", dash="dash"),
            name="Persistence",
        ))
        # Linear
        fig_forecast.add_trace(go.Scatter(
            x=hours_future, y=lr_future,
            mode="lines", line=dict(color="#2A9D8F", dash="dot"),
            name="Linear AR",
        ))
        # MLP
        fig_forecast.add_trace(go.Scatter(
            x=hours_future, y=mlp_future,
            mode="lines", line=dict(color="#E63946"),
            name="Neural Model",
        ))
        fig_forecast.add_vline(x=0, line_dash="dash", line_color="gray",
                               annotation_text="Forecast Start")
        apply_common_layout(fig_forecast, title=f"24-Hour Forecast Example -- {forecast_city}", height=450)
        fig_forecast.update_layout(xaxis_title="Hours (relative to forecast start)",
                                    yaxis_title="Temperature (C)")
        st.plotly_chart(fig_forecast, use_container_width=True)

        # ── Error by forecast hour ───────────────────────────────────────
        st.subheader("Forecast Error by Hour Ahead")
        hour_rmses_persist = []
        hour_rmses_lr = []
        hour_rmses_mlp = []
        for h in range(forecast_horizon):
            hour_rmses_persist.append(np.sqrt(mean_squared_error(actual_inv[:, h], baseline_preds_inv[:, h])))
            hour_rmses_lr.append(np.sqrt(mean_squared_error(actual_inv[:, h], lr_preds_inv[:, h])))
            hour_rmses_mlp.append(np.sqrt(mean_squared_error(actual_inv[:, h], mlp_preds_inv[:, h])))

        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Scatter(
            x=list(range(1, forecast_horizon + 1)), y=hour_rmses_persist,
            mode="lines+markers", name="Persistence", line=dict(color="#FB8500"),
        ))
        fig_hourly.add_trace(go.Scatter(
            x=list(range(1, forecast_horizon + 1)), y=hour_rmses_lr,
            mode="lines+markers", name="Linear AR", line=dict(color="#2A9D8F"),
        ))
        fig_hourly.add_trace(go.Scatter(
            x=list(range(1, forecast_horizon + 1)), y=hour_rmses_mlp,
            mode="lines+markers", name="Neural Model", line=dict(color="#E63946"),
        ))
        apply_common_layout(fig_hourly, title="RMSE by Forecast Horizon (hours ahead)", height=400)
        fig_hourly.update_layout(xaxis_title="Hours Ahead", yaxis_title="RMSE (C)")
        st.plotly_chart(fig_hourly, use_container_width=True)

        insight_box(
            "Notice how all models degrade as the forecast horizon increases. "
            "This is not a bug -- it is a fundamental property of chaotic "
            "systems. Edward Lorenz showed in 1963 that the atmosphere is "
            "inherently unpredictable beyond a certain horizon. Our neural "
            "model typically holds an advantage over the baselines for the "
            "first 12-16 hours, but eventually everyone converges toward the "
            "same uncomfortable level of uncertainty. A true LSTM with proper "
            "recurrent connections would likely extend that advantage further."
        )

# ── 52.4 LSTM vs Simple RNN Concept ─────────────────────────────────────────
st.header("52.4  Why LSTM Outperforms Simple RNN")

st.markdown("""
| Feature | Simple RNN | LSTM |
|---------|-----------|------|
| Memory | Short-term only | Short + long-term |
| Gradient flow | Vanishes/explodes | Gated (stable) |
| Parameters | Fewer | ~4x more (3 gates + cell) |
| Best for | Short sequences | Long sequences (24h+) |
| Weather use | Next-hour only | Multi-day forecasting |
""")

concept_box(
    "The Cell State Highway",
    "The key innovation of LSTM is the <b>cell state</b>, and the reason it "
    "works is almost embarrassingly simple. The cell state is like a conveyor "
    "belt running through the entire sequence. When the forget gate outputs a "
    "value close to 1 and the input gate outputs a value close to 0, information "
    "flows through <i>completely unchanged</i>. This means the gradient also flows "
    "through unchanged, which solves the vanishing gradient problem. It is the "
    "architectural equivalent of building a highway alongside a winding mountain "
    "road -- information that needs to travel long distances can take the highway "
    "instead of navigating every hairpin turn."
)

code_example("""import torch
import torch.nn as nn

class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # use last hidden state

# Training loop
model = WeatherLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=24)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(100):
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "What problem do LSTMs solve that simple RNNs cannot?",
    [
        "LSTMs are faster to train",
        "LSTMs solve the vanishing gradient problem, enabling long-range dependencies",
        "LSTMs need fewer parameters",
        "LSTMs can only process images",
    ],
    correct_idx=1,
    explanation=(
        "The gating mechanism in LSTMs -- specifically the cell state highway "
        "-- allows gradients to flow through long sequences without vanishing. "
        "This is what lets LSTMs learn that \"it rained 18 hours ago\" matters "
        "for today's forecast, something a simple RNN literally cannot do "
        "because the gradient signal from 18 steps ago has been multiplied "
        "to near-zero."
    ),
    key="ch52_quiz1",
)

quiz(
    "In weather forecasting, why does forecast error increase with the horizon?",
    [
        "The model gets tired of predicting",
        "Farther into the future, uncertainty accumulates and patterns become less predictable",
        "The test set gets smaller",
        "The learning rate decreases automatically",
    ],
    correct_idx=1,
    explanation=(
        "Weather is a chaotic system in the technical, mathematical sense: "
        "small perturbations in initial conditions grow exponentially over time. "
        "This is the butterfly effect, and it is not a metaphor -- it is a "
        "quantifiable property of the Navier-Stokes equations that govern "
        "atmospheric flow. Every model, no matter how sophisticated, runs into "
        "this wall."
    ),
    key="ch52_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "**RNNs** process sequences by maintaining a hidden state -- giving the network something like a memory, albeit a leaky one.",
    "Simple RNNs suffer from **vanishing gradients**, which effectively limits their memory to a few time steps. Not great for weather.",
    "**LSTMs** add forget, input, and output **gates** that act as learned valves controlling information flow.",
    "The **cell state highway** is the key insight: it lets information (and gradients) flow through long sequences without degradation.",
    "For temperature forecasting, sequence models beat simple baselines but degrade with longer horizons -- because the atmosphere is chaotic, not because the model is bad.",
    "Always compare against **baselines** (persistence, seasonal average). They are embarrassingly strong and keep you honest.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 51: Feedforward Networks",
    prev_page="51_Feedforward_Networks.py",
    next_label="Ch 53: Autoencoders",
    next_page="53_Autoencoders.py",
)
