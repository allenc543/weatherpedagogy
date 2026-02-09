"""Chapter 51: Feedforward Networks -- Hidden layers, dropout, batch norm, architecture builder."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.ml_helpers import prepare_classification_data, classification_metrics, plot_confusion_matrix
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box, warning_box,
    code_example, quiz, takeaways, navigation,
)

# ── Header ───────────────────────────────────────────────────────────────────
chapter_header(51, "Feedforward Networks", part="XII")
st.markdown(
    "Last chapter we established that a single neuron is just linear regression "
    "in a fancy hat. The natural next question: what happens when you give the "
    "hat more hats? A feedforward network (also called a Multi-Layer Perceptron "
    "or MLP) stacks multiple layers of these neurons, and the result is a system "
    "that can learn genuinely non-linear decision boundaries. This chapter lets "
    "you **build your own architecture** and pit it against a Random Forest for "
    "city classification."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 51.1 Architecture Concepts ──────────────────────────────────────────────
st.header("51.1  Feedforward Network Architecture")

concept_box(
    "Layers and Neurons",
    "A feedforward network is organized into layers, and data flows strictly "
    "forward (no loops, no looking back -- that is what RNNs are for):<br>"
    "- <b>Input layer</b>: one neuron per feature. We have 4 weather features, "
    "so 4 input neurons.<br>"
    "- <b>Hidden layers</b>: where the actual learning happens. More layers and "
    "more neurons per layer means more <i>capacity</i> -- the network can "
    "represent more complex functions. Of course, more capacity also means "
    "more opportunities to memorize noise instead of signal.<br>"
    "- <b>Output layer</b>: one neuron per class (6 cities), with softmax to turn "
    "raw scores into probabilities."
)

formula_box(
    "Forward Pass",
    r"\mathbf{h}^{(l)} = \sigma\!\left(\mathbf{W}^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right)",
    "Each layer transforms the previous layer's output through weights W, bias b, "
    "and activation sigma. Stack enough of these and you can approximate any "
    "continuous function. (This is literally a theorem, not hype.)"
)

st.markdown("""
**Regularization techniques** (or: how to stop your network from memorizing the training set):
- **Dropout**: randomly zero out neurons during training. This sounds like sabotage, but it forces the network to develop redundant representations instead of relying on any single fragile pathway. It is the neural network equivalent of cross-training.
- **Early stopping**: stop training when validation loss stops improving. Just because you *can* train for 10,000 epochs does not mean you *should*.
- **L2 regularization (weight decay)**: penalize large weights. Big weights mean the model is screaming about small differences in input, which is usually a sign of overfitting.
""")

# ── 51.2 Prepare Data ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, le, scaler = prepare_classification_data(
    fdf, FEATURE_COLS, target="city", test_size=0.2
)
city_labels = le.classes_

# Subsample for speed
sample_n = min(6000, len(X_train))
rng = np.random.RandomState(42)
idx = rng.choice(len(X_train), sample_n, replace=False)
X_tr_s = X_train.iloc[idx]
y_tr_s = y_train[idx]

# ── 51.3 Interactive Architecture Builder ────────────────────────────────────
st.header("51.2  Interactive Architecture Builder")

st.markdown(
    "Here is where it gets fun. Design your own feedforward network below and "
    "see how it does on city classification. Fair warning: there is no substitute "
    "for just trying things. The field has rough heuristics but no guaranteed "
    "recipes."
)

col1, col2 = st.columns(2)
with col1:
    n_layers = st.slider("Number of hidden layers", 1, 5, 2, key="ff_layers")
with col2:
    activation = st.selectbox("Activation function", ["relu", "tanh", "logistic"], key="ff_act")

layer_sizes = []
st.markdown("**Neurons per layer:**")
layer_cols = st.columns(min(n_layers, 5))
for i in range(n_layers):
    with layer_cols[i % 5]:
        size = st.number_input(
            f"Layer {i+1}", min_value=2, max_value=128, value=max(32 // (i+1), 8),
            step=4, key=f"layer_{i}",
        )
        layer_sizes.append(size)

col_a, col_b = st.columns(2)
with col_a:
    alpha = st.select_slider(
        "L2 Regularization (alpha)",
        options=[0.0001, 0.001, 0.01, 0.1],
        value=0.001,
        key="ff_alpha",
    )
with col_b:
    max_iter = st.slider("Max training epochs", 100, 1000, 300, 50, key="ff_epochs")

# Architecture visualization
st.subheader("Network Architecture")
arch_text = f"Input ({len(FEATURE_COLS)} features)"
for i, s in enumerate(layer_sizes):
    arch_text += f"  -->  Hidden{i+1} ({s} neurons, {activation})"
arch_text += f"  -->  Output ({len(city_labels)} cities, softmax)"
st.code(arch_text, language=None)

total_params = 0
prev_size = len(FEATURE_COLS)
for s in layer_sizes:
    total_params += prev_size * s + s  # weights + biases
    prev_size = s
total_params += prev_size * len(city_labels) + len(city_labels)  # output layer
st.info(
    f"Total trainable parameters: **{total_params:,}**. "
    f"Our network has {total_params:,} parameters to distinguish between "
    f"{len(city_labels)} cities. The actual atmosphere has ~10^44 molecules. "
    f"We are being parsimonious."
)

# Train the network
with st.spinner("Training feedforward network..."):
    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(layer_sizes),
        activation=activation,
        alpha=alpha,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        learning_rate="adaptive",
        learning_rate_init=0.001,
    )
    mlp.fit(X_tr_s, y_tr_s)

mlp_train_acc = accuracy_score(y_tr_s, mlp.predict(X_tr_s))
mlp_test_acc = accuracy_score(y_test, mlp.predict(X_test))

c1, c2, c3 = st.columns(3)
c1.metric("Train Accuracy", f"{mlp_train_acc:.4f}")
c2.metric("Test Accuracy", f"{mlp_test_acc:.4f}")
c3.metric("Training Epochs", f"{mlp.n_iter_}")

# Loss curve
if hasattr(mlp, 'loss_curve_'):
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=list(range(1, len(mlp.loss_curve_) + 1)), y=mlp.loss_curve_,
        mode="lines", line=dict(color="#E63946"),
        name="Training Loss",
    ))
    if hasattr(mlp, 'validation_scores_') and mlp.validation_scores_:
        fig_loss.add_trace(go.Scatter(
            x=list(range(1, len(mlp.validation_scores_) + 1)),
            y=[1 - s for s in mlp.validation_scores_],
            mode="lines", line=dict(color="#2A9D8F"),
            name="Validation Error (1 - acc)",
        ))
    apply_common_layout(fig_loss, title="Training Loss Curve", height=400)
    fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig_loss, use_container_width=True)

# ── 51.4 Compare to Random Forest ───────────────────────────────────────────
st.header("51.3  Neural Network vs Random Forest")

rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_tr_s, y_tr_s)
rf_train_acc = accuracy_score(y_tr_s, rf.predict(X_tr_s))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))

comp_df = pd.DataFrame({
    "Model": [f"Feedforward NN ({n_layers} layers)", "Random Forest (100 trees)"],
    "Train Accuracy": [mlp_train_acc, rf_train_acc],
    "Test Accuracy": [mlp_test_acc, rf_test_acc],
    "Gap (Overfit)": [mlp_train_acc - mlp_test_acc, rf_train_acc - rf_test_acc],
})
st.dataframe(
    comp_df.style.format({
        "Train Accuracy": "{:.4f}", "Test Accuracy": "{:.4f}", "Gap (Overfit)": "{:.4f}",
    }).highlight_max(subset=["Test Accuracy"], color="#d4edda"),
    use_container_width=True, hide_index=True,
)

fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(x=comp_df["Model"], y=comp_df["Train Accuracy"],
                          name="Train", marker_color="#264653"))
fig_comp.add_trace(go.Bar(x=comp_df["Model"], y=comp_df["Test Accuracy"],
                          name="Test", marker_color="#E63946"))
apply_common_layout(fig_comp, title="NN vs Random Forest", height=400)
fig_comp.update_layout(barmode="group", yaxis_title="Accuracy")
st.plotly_chart(fig_comp, use_container_width=True)

if mlp_test_acc > rf_test_acc:
    st.success(f"Your neural network outperforms Random Forest by {(mlp_test_acc - rf_test_acc)*100:.2f} pp!")
elif rf_test_acc > mlp_test_acc:
    st.info(
        f"Random Forest outperforms your NN by {(rf_test_acc - mlp_test_acc)*100:.2f} pp. "
        "This is not a moral failing. For tabular data with a handful of features, "
        "tree-based models are often competitive or better. Neural networks tend to "
        "earn their keep on unstructured data -- images, text, audio. Try tweaking "
        "architecture, epochs, or regularization and see if you can close the gap."
    )
else:
    st.info("Both models perform similarly -- which honestly is the most common outcome for tabular data.")

# ── 51.5 Confusion Matrices Side by Side ────────────────────────────────────
st.header("51.4  Confusion Matrices")
col_cm1, col_cm2 = st.columns(2)

with col_cm1:
    st.subheader("Feedforward NN")
    y_mlp_pred = mlp.predict(X_test)
    m_mlp = classification_metrics(y_test, y_mlp_pred, labels=city_labels)
    fig_cm_mlp = plot_confusion_matrix(m_mlp["confusion_matrix"], city_labels)
    fig_cm_mlp.update_layout(height=400)
    st.plotly_chart(fig_cm_mlp, use_container_width=True)

with col_cm2:
    st.subheader("Random Forest")
    y_rf_pred = rf.predict(X_test)
    m_rf = classification_metrics(y_test, y_rf_pred, labels=city_labels)
    fig_cm_rf = plot_confusion_matrix(m_rf["confusion_matrix"], city_labels)
    fig_cm_rf.update_layout(height=400)
    st.plotly_chart(fig_cm_rf, use_container_width=True)

# ── 51.6 Effect of Architecture Depth ────────────────────────────────────────
st.header("51.5  Effect of Network Depth and Width")

st.markdown(
    "You might ask: if more neurons means more capacity, why not just make "
    "the network enormous? Let us find out what actually happens."
)

architectures = {
    "(16,)": (16,),
    "(32, 16)": (32, 16),
    "(64, 32, 16)": (64, 32, 16),
    "(128, 64, 32, 16)": (128, 64, 32, 16),
    "(64, 64, 64, 64, 64)": (64, 64, 64, 64, 64),
}

arch_results = []
with st.spinner("Training architectures for comparison..."):
    for name, layers in architectures.items():
        m = MLPClassifier(
            hidden_layer_sizes=layers, activation="relu",
            alpha=0.001, max_iter=300, random_state=42,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=15, learning_rate_init=0.001,
        )
        m.fit(X_tr_s, y_tr_s)
        n_p = 0
        prev = len(FEATURE_COLS)
        for s in layers:
            n_p += prev * s + s
            prev = s
        n_p += prev * len(city_labels) + len(city_labels)
        arch_results.append({
            "Architecture": name,
            "Layers": len(layers),
            "Parameters": n_p,
            "Train Acc": accuracy_score(y_tr_s, m.predict(X_tr_s)),
            "Test Acc": accuracy_score(y_test, m.predict(X_test)),
            "Epochs": m.n_iter_,
        })

arch_df = pd.DataFrame(arch_results)
st.dataframe(
    arch_df.style.format({
        "Parameters": "{:,}", "Train Acc": "{:.4f}",
        "Test Acc": "{:.4f}", "Epochs": "{:d}",
    }).highlight_max(subset=["Test Acc"], color="#d4edda"),
    use_container_width=True, hide_index=True,
)

fig_arch = go.Figure()
fig_arch.add_trace(go.Scatter(
    x=arch_df["Parameters"], y=arch_df["Train Acc"],
    mode="lines+markers+text", name="Train",
    text=arch_df["Architecture"], textposition="top center",
    line=dict(color="#264653"),
))
fig_arch.add_trace(go.Scatter(
    x=arch_df["Parameters"], y=arch_df["Test Acc"],
    mode="lines+markers", name="Test",
    line=dict(color="#E63946"),
))
apply_common_layout(fig_arch, title="Accuracy vs Model Size (Parameters)", height=450)
fig_arch.update_layout(xaxis_title="Number of Parameters", yaxis_title="Accuracy")
st.plotly_chart(fig_arch, use_container_width=True)

insight_box(
    "And here is the punchline: more parameters does not always mean better "
    "test accuracy. The train-test gap tends to widen as the network grows, "
    "which is overfitting in action. For this tabular dataset with only 4 "
    "features, a moderately-sized network is usually the sweet spot. The "
    "enormous networks are like bringing a battleship to a pond -- impressive, "
    "but not obviously useful."
)

code_example("""from sklearn.neural_network import MLPClassifier

# Feedforward network with dropout via early stopping
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    alpha=0.001,           # L2 regularization
    max_iter=500,
    early_stopping=True,   # monitor validation loss
    validation_fraction=0.15,
    n_iter_no_change=15,   # patience
    learning_rate='adaptive',
    random_state=42,
)
mlp.fit(X_train_scaled, y_train)
print(f"Stopped at epoch {mlp.n_iter_}")
print(f"Test accuracy: {mlp.score(X_test_scaled, y_test):.4f}")
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "What is the purpose of dropout in a neural network?",
    [
        "To speed up training",
        "To randomly zero out neurons during training, preventing co-adaptation",
        "To add more neurons to the network",
        "To normalize the inputs",
    ],
    correct_idx=1,
    explanation=(
        "Dropout randomly kills neurons during training -- which sounds "
        "destructive, and it is, by design. It forces the network to develop "
        "redundant representations rather than building a fragile house of "
        "cards where everything depends on a few key neurons. Think of it as "
        "the neural network version of not putting all your eggs in one basket."
    ),
    key="ch51_quiz1",
)

quiz(
    "For tabular data with 4 features and 6 classes, neural networks typically:",
    [
        "Always outperform tree-based methods",
        "Are competitive but tree-based methods are often equally good or better",
        "Cannot be used because there are too few features",
        "Require at least 10 hidden layers",
    ],
    correct_idx=1,
    explanation=(
        "This is one of the dirty secrets of applied ML: for tabular data, "
        "tree-based methods (Random Forest, XGBoost) are stubbornly competitive "
        "with neural networks. NNs really shine on unstructured data -- images, "
        "text, audio, sequences -- where the raw inputs need hierarchical "
        "feature extraction. On a spreadsheet with 4 columns, gradient-boosted "
        "trees are hard to beat."
    ),
    key="ch51_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "Feedforward networks stack **hidden layers** of neurons with non-linear activations to learn complex decision boundaries.",
    "More layers and neurons increase **capacity** but also overfitting risk -- there is no free lunch here.",
    "**Regularization** (dropout, L2, early stopping) is not optional overhead; it is what prevents your network from memorizing noise.",
    "For tabular weather data with a handful of features, tree-based models are often just as good. Neural nets earn their keep on unstructured data.",
    "The **loss curve** is your diagnostic dashboard: divergence means your learning rate is too high, a flat line means it is too low, and a widening train-test gap means overfitting.",
    "Network architecture is a hyperparameter, not a science -- experiment with different depths and widths, and let the validation set be the judge.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 50: Neural Network Basics",
    prev_page="50_Neural_Network_Basics.py",
    next_label="Ch 52: RNN & LSTM",
    next_page="52_RNN_and_LSTM.py",
)
