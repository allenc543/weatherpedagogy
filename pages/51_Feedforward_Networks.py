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
    "Let me set up the specific problem we are going to solve, because neural network "
    "jargon only makes sense when you can see what it does to actual data."
)
st.markdown(
    "**The task**: You receive a single weather reading -- temperature (22.5 C), "
    "relative humidity (65%), wind speed (12 km/h), and surface pressure (1013 hPa) -- "
    "and you need to guess which of our 6 cities (Dallas, Houston, Austin, San Antonio, "
    "NYC, or Los Angeles) it came from. Four numbers in, one city name out. "
    "Last chapter, we saw that a single neuron is basically linear regression wearing "
    "a Halloween costume: it can only draw straight-line boundaries between cities. "
    "That is fine if Dallas and NYC live on opposite sides of a neat dividing line in "
    "temperature-humidity space, but it falls apart when the cities overlap in "
    "complicated ways -- and they do."
)
st.markdown(
    "This chapter is about what happens when you **stack multiple layers** of neurons. "
    "The result -- called a feedforward network or Multi-Layer Perceptron (MLP) -- "
    "can learn curved, twisted, and wonderfully non-linear decision boundaries. You will "
    "**build your own network architecture** and pit it against a Random Forest on our "
    "city classification task. Spoiler: the results might surprise you."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 51.1 Architecture Concepts ──────────────────────────────────────────────
st.header("51.1  Feedforward Network Architecture")

st.markdown(
    "Before I throw the terms 'hidden layer' and 'activation function' at you, "
    "let me walk through what actually happens when our network processes a weather reading."
)
st.markdown(
    "Imagine your 4-number weather reading (22.5 C, 65%, 12 km/h, 1013 hPa) walks "
    "into a building. The building has multiple floors. On the ground floor, 4 clerks "
    "each receive one number. They pass their numbers up to the second floor, where, "
    "say, 32 clerks each look at all 4 numbers, do some math, and produce their own "
    "single number. Those 32 numbers go up to the third floor, where 16 clerks do the "
    "same thing. Finally, the top floor has 6 clerks -- one per city -- who each "
    "produce a confidence score. The city with the highest score wins."
)
st.markdown(
    "The crucial rule: information only flows **upward**. No floor sends messages back "
    "down. No floor peeks at what the floors above it are doing. This is why it is "
    "called 'feedforward' -- data goes in one direction, from input to output, and "
    "that is that. (Networks that *do* look backward are called recurrent networks, "
    "and we will meet them in Chapter 52.)"
)

concept_box(
    "Layers and Neurons",
    "Now that you have the picture, here are the official names:<br><br>"
    "- <b>Input layer</b>: the ground floor. One neuron per feature. We have 4 weather "
    "features, so 4 input neurons. These do not do any computation -- they just pass "
    "the raw numbers along.<br>"
    "- <b>Hidden layers</b>: the middle floors where the actual learning happens. "
    "'Hidden' just means they are not directly visible as inputs or outputs. Each "
    "neuron takes all the numbers from the previous layer, multiplies them by learned "
    "weights, adds a bias, and applies a non-linear activation function. More layers "
    "and more neurons per layer means more <i>capacity</i> -- the network can represent "
    "more complex boundaries between cities. But more capacity also means more "
    "opportunity to memorize noise, exactly like raising the polynomial degree in "
    "Chapter 44.<br>"
    "- <b>Output layer</b>: the top floor. One neuron per class (6 cities), with "
    "softmax to turn raw scores into probabilities that sum to 1. So the output "
    "might be: Dallas 4%, Houston 62%, Austin 18%, San Antonio 11%, NYC 3%, LA 2% -- "
    "and we predict Houston."
)

formula_box(
    "Forward Pass",
    r"\mathbf{h}^{(l)} = \sigma\!\left(\mathbf{W}^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right)",
    "Each layer transforms the previous layer's output through weights W, bias b, "
    "and activation function sigma. In plain English: layer 2 takes the 4 numbers "
    "from layer 1, multiplies each by a learned weight, adds them up, adds a bias "
    "term, and then applies a non-linear squish (like ReLU, which just sets negative "
    "values to zero). Stack enough of these transformations and you can approximate "
    "any continuous function -- this is literally a theorem (the Universal "
    "Approximation Theorem), not hype."
)

st.markdown("""
**Regularization techniques** (or: how to stop your network from memorizing the 105,264 training rows):

- **Dropout**: During each training step, randomly zero out some neurons -- say, 20% of them. This sounds like sabotage, and it is, *by design*. It forces the network to develop redundant pathways for distinguishing cities. If neuron #7 on layer 2 is the only one that learned "low humidity + warm = probably LA," and we randomly kill neuron #7, the network has to spread that knowledge across other neurons too. Think of it as cross-training: no single neuron becomes a bottleneck.

- **Early stopping**: Stop training when validation loss stops improving. Our network could keep training for 10,000 epochs, driving training accuracy from 85% to 99.9%, but that last 15% of improvement would come entirely from memorizing specific weather readings, not learning generalizable patterns. When the validation error starts climbing while training error keeps dropping -- that is your cue to stop.

- **L2 regularization (weight decay)**: Penalize large weights. When a weight is huge, it means the network is screaming about tiny input differences -- "the humidity was 65.1% instead of 65.0%? COMPLETELY DIFFERENT CITY!" Large weights are the neural network equivalent of overreacting to noise. L2 penalizes this, keeping the weights modest and the network calm.
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
    "Here is where you become the architect. Design your own feedforward network "
    "below and see how well it can distinguish between 6 cities from 4 weather "
    "features. There is no guaranteed recipe -- the field has rough heuristics "
    "('start with 1-2 hidden layers,' 'use ReLU unless you have a reason not to') "
    "but the honest truth is that architecture design is part science, part "
    "craft, part just trying things."
)
st.markdown(
    "**What the controls do:** The 'number of hidden layers' slider sets how many "
    "middle floors your building has. 'Neurons per layer' sets how many clerks "
    "work on each floor. 'Activation function' determines what non-linear squish "
    "each neuron applies (ReLU is almost always the right choice for hidden layers). "
    "'L2 regularization' controls how harshly large weights are penalized. 'Max "
    "training epochs' is the maximum number of full passes through the training data."
)
st.markdown(
    "**What to look for:** Pay attention to the gap between train and test accuracy. "
    "A big gap (e.g., train 95%, test 78%) means overfitting -- your network memorized "
    "weather readings instead of learning city patterns. Both being low (e.g., 55% each) "
    "means underfitting -- the network is too small. The sweet spot is high test accuracy "
    "with a small train-test gap."
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
    f"That is {total_params:,} individual numbers the network will learn by adjusting them "
    f"to minimize classification error. For context: we are using these {total_params:,} "
    f"numbers to distinguish between {len(city_labels)} cities using 4 weather features. "
    f"The actual atmosphere has roughly 10^44 molecules. We are being parsimonious."
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

st.markdown(
    f"Your network trained for {mlp.n_iter_} epochs before early stopping kicked in. "
    f"Train accuracy: {mlp_train_acc:.1%}. Test accuracy: {mlp_test_acc:.1%}. "
    f"The gap between them ({(mlp_train_acc - mlp_test_acc)*100:.1f} percentage points) "
    f"tells you how much the network is overfitting. A gap under 5 pp is usually fine. "
    f"A gap over 15 pp means the network is memorizing specific weather readings "
    f"rather than learning what makes Dallas different from Houston."
)

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

st.markdown(
    "**How to read the loss curve:** The red line should drop steeply at first (the "
    "network is learning the big patterns -- 'NYC is colder than Houston') and then "
    "flatten out (it has learned the main structure and is fine-tuning). If the red "
    "line oscillates wildly, the learning rate is too high. If it barely moves, the "
    "learning rate is too low. If the green validation line starts climbing while the "
    "red line keeps dropping, the network has started memorizing training data -- and "
    "that is when early stopping should kick in."
)

# ── 51.4 Compare to Random Forest ───────────────────────────────────────────
st.header("51.3  Neural Network vs Random Forest")

st.markdown(
    "Here is a question that matters in practice: is a neural network actually "
    "*better* than a Random Forest for this task? Remember, we have 4 numerical "
    "features and 6 classes. This is textbook tabular data -- the kind of problem "
    "tree-based methods were born for. Let us put both models on equal footing "
    "and see what happens."
)

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
        "This is not a moral failing -- it is one of the dirty secrets of applied ML. "
        "For tabular data with a handful of numerical columns (like our 4 weather features), "
        "tree-based models (Random Forest, XGBoost, LightGBM) are stubbornly competitive "
        "with neural networks. Neural nets earn their keep on *unstructured* data -- images, "
        "text, audio, long sequences -- where the raw inputs need hierarchical feature "
        "extraction. On a spreadsheet with 4 columns? Trees are hard to beat. Try tweaking "
        "your architecture, regularization, or epoch count and see if you can close the gap."
    )
else:
    st.info("Both models perform similarly -- which honestly is the most common outcome for tabular data of this kind.")

# ── 51.5 Confusion Matrices Side by Side ────────────────────────────────────
st.header("51.4  Confusion Matrices")

st.markdown(
    "Accuracy is a single number. The confusion matrix shows you *where* each model "
    "gets confused. Does the neural network mix up Dallas and Austin (both hot Texas "
    "cities with similar weather)? Does it correctly separate NYC (cold, humid winters) "
    "from LA (mild, dry)? Compare the two matrices side by side."
)

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

st.markdown(
    "**Reading the matrix:** Each row is the true city, each column is the predicted city. "
    "Bright diagonal = correct predictions. Off-diagonal entries = confusion. If the "
    "Dallas-Austin cell is bright, it means the model frequently mistakes one for the "
    "other -- understandable, since both are Texas cities with similar climates. "
    "NYC and LA should be easy to separate (completely different weather profiles), "
    "so check that those off-diagonal cells are dark."
)

# ── 51.6 Effect of Architecture Depth ────────────────────────────────────────
st.header("51.5  Effect of Network Depth and Width")

st.markdown(
    "You might ask: 'If more neurons means more capacity to learn complex city "
    "boundaries, why not just make the network enormous?' This is the exact same "
    "question as 'why not use a degree-50 polynomial?' from Chapter 44, and the "
    "answer is the same: overfitting. Let us run the experiment and see."
)
st.markdown(
    "Below, we train 5 different architectures on the same city classification task, "
    "ranging from a tiny 1-layer network with 16 neurons to a 5-layer monster with "
    "64 neurons per layer. Watch what happens to the train-test gap as the network "
    "grows."
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
    "And here is the punchline, the same one from Chapter 44 but wearing a neural "
    "network hat: more parameters does not always mean better test accuracy. The "
    "train-test gap tends to widen as the network grows, which is overfitting in "
    "action. For this tabular dataset -- 4 weather features, 6 cities, ~105,000 "
    "rows -- a moderately-sized network (64-32-16 or even 32-16) is usually the "
    "sweet spot. The 5-layer network with 64 neurons per layer has over 12,000 "
    "parameters trying to learn what amounts to a few curved boundaries in 4D "
    "space. That is like bringing a battleship to a pond -- impressive, but not "
    "obviously useful."
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
        "Dropout randomly kills neurons during each training step. Say neuron #12 in "
        "layer 2 has become the sole expert on 'low humidity means LA.' If we randomly "
        "zero it out, the network is forced to spread that knowledge to other neurons "
        "too -- maybe neurons #5 and #23 learn partial versions of the same rule. This "
        "redundancy makes the network much more robust. It is the neural network version "
        "of not putting all your eggs in one basket. Without dropout, networks tend to "
        "develop fragile, co-dependent pathways where everything hinges on a few key "
        "neurons. With dropout, no single neuron is indispensable."
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
        "This is one of the most important practical lessons in applied ML. For tabular "
        "data -- rows and columns, numerical features, the kind of thing that lives in a "
        "spreadsheet or CSV file -- tree-based methods (Random Forest, XGBoost, LightGBM) "
        "are stubbornly competitive with neural networks. In benchmarks on hundreds of "
        "tabular datasets, tree methods win roughly as often as neural nets. Neural networks "
        "really shine on unstructured data: images (pixels need hierarchical feature extraction), "
        "text (sequences need contextual understanding), audio, video. On our weather dataset "
        "with 4 columns (temperature, humidity, wind, pressure), a well-tuned Random Forest "
        "is a formidable competitor. That does not make the neural network useless -- it "
        "means you should always benchmark against a tree-based baseline."
    ),
    key="ch51_quiz2",
)

quiz(
    "Your feedforward network achieves 97% training accuracy but only 72% test accuracy "
    "on city classification. What is the most likely issue and the best fix?",
    [
        "High bias -- increase the number of layers and neurons",
        "High variance (overfitting) -- add regularization, reduce network size, or get more data",
        "The learning rate is too low -- increase it",
        "The data is too noisy to model",
    ],
    correct_idx=1,
    explanation=(
        "A 25 percentage point gap between training and test accuracy is the textbook "
        "signature of overfitting (high variance). The network has memorized specific "
        "weather readings from the training set -- it 'knows' that a reading of exactly "
        "22.47 C / 64.8% / 11.3 km/h / 1013.2 hPa is from Houston because it saw that "
        "exact row during training. But it has not learned the general pattern of what "
        "makes Houston weather different from Dallas weather. The fixes, in order of what "
        "to try first: (1) increase L2 regularization (alpha), (2) reduce the network size "
        "(fewer layers or neurons), (3) enable dropout or early stopping, (4) get more "
        "training data. Adding MORE capacity (more layers) would make this worse, not better."
    ),
    key="ch51_quiz3",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "A feedforward network stacks **hidden layers** of neurons to learn non-linear decision "
    "boundaries between cities. Our 4 weather features (temp, humidity, wind, pressure) "
    "pass through these layers and produce a probability for each of 6 cities.",
    "More layers and neurons increase **capacity** (the network can represent more complex "
    "boundaries) but also overfitting risk. A (64, 32, 16) network with 4 weather inputs "
    "has thousands of parameters -- plenty for this task.",
    "**Regularization** (dropout, L2, early stopping) prevents the network from memorizing "
    "that 'this specific humidity reading at this specific temperature was from Austin' "
    "instead of learning 'Austin tends to be hot and moderately humid.'",
    "For tabular weather data with a handful of features, tree-based models (Random Forest, "
    "XGBoost) are often just as good as neural nets. Always benchmark. Neural nets earn "
    "their keep on images, text, and sequences.",
    "The **loss curve** is your diagnostic dashboard: a steep initial drop means the network "
    "is learning real patterns, a flat line means the learning rate needs adjustment, and a "
    "divergence between training and validation curves means overfitting has begun.",
    "Network architecture is a hyperparameter, not a science. Start with 1-2 layers of "
    "32-64 neurons, use ReLU, and let the validation set be the judge.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 50: Neural Network Basics",
    prev_page="50_Neural_Network_Basics.py",
    next_label="Ch 52: RNN & LSTM",
    next_page="52_RNN_and_LSTM.py",
)
