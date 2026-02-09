"""Chapter 49: Stacking -- Meta-learner, model diversity, ensemble combination."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
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
chapter_header(49, "Stacking (Stacked Generalization)", part="XI")
st.markdown(
    "Stacking is the 'throw everything at the wall' approach to machine learning, "
    "except it actually works. Instead of picking one model and hoping for the best, "
    "you train several different models and then train a **meta-learner** that figures "
    "out the optimal way to combine their predictions. It is not just voting -- it is "
    "weighted, learned, context-dependent combination. Think of it as hiring a team "
    "of specialists and then hiring a manager whose only job is to decide which "
    "specialist to listen to in each situation."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ── 49.1 How Stacking Works ─────────────────────────────────────────────────
st.header("49.1  How Stacking Works")

concept_box(
    "Two-Level Architecture",
    "<b>Level 0 (Base Learners)</b>: Train several diverse models -- logistic regression, "
    "random forest, gradient boosting, KNN, whatever you want. Each generates predictions "
    "using cross-validation (so the meta-learner never sees predictions on data that was "
    "used for training, which would be leakage).<br><br>"
    "<b>Level 1 (Meta-Learner)</b>: A simple model (often logistic regression -- you want "
    "this to be simple so it does not overfit to the base model quirks) is trained on the "
    "base learners' predictions. Its job is to figure out: 'When the random forest says "
    "Dallas and the logistic regression says San Antonio, who should I believe?'"
)

formula_box(
    "Stacking Prediction",
    r"\hat{y} = g\!\left(\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_M\right)",
    "g() is the meta-learner; y_hat_1..M are base model predictions. The meta-learner learns how to weigh and combine them optimally."
)

st.markdown("""
**Why stacking works (when it works):**
1. **Model diversity**: Different algorithms see the data differently -- linear models, tree models, and distance-based models all have different blind spots.
2. **Error cancellation**: Where one model fails, another might succeed. The meta-learner can learn to trust different models in different situations.
3. **Learned combination**: Unlike simple voting, stacking does not treat all models equally. It gives more weight to models that are reliable and less to those that are not.
""")

# ── 49.2 Prepare Data ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, le, scaler = prepare_classification_data(
    fdf, FEATURE_COLS, target="city", test_size=0.2
)
city_labels = le.classes_

# Subsample for speed
sample_n = min(5000, len(X_train))
rng = np.random.RandomState(42)
idx = rng.choice(len(X_train), sample_n, replace=False)
X_tr_s = X_train.iloc[idx]
y_tr_s = y_train[idx]

# ── 49.3 Individual Base Models ──────────────────────────────────────────────
st.header("49.2  Base Learner Performance")

available_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
}

st.markdown("First, let us see how each model does on its own. The question is whether combining them can beat the best individual:")

individual_results = []
for name, model in available_models.items():
    model.fit(X_tr_s, y_tr_s)
    train_acc = accuracy_score(y_tr_s, model.predict(X_tr_s))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    cv_scores = cross_val_score(model, X_tr_s, y_tr_s, cv=5, scoring="accuracy")
    individual_results.append({
        "Model": name,
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "CV Mean": cv_scores.mean(),
        "CV Std": cv_scores.std(),
    })

indiv_df = pd.DataFrame(individual_results).sort_values("Test Accuracy", ascending=False)
st.dataframe(
    indiv_df.style.format({
        "Train Accuracy": "{:.4f}", "Test Accuracy": "{:.4f}",
        "CV Mean": "{:.4f}", "CV Std": "{:.4f}",
    }).highlight_max(subset=["Test Accuracy"], color="#d4edda"),
    use_container_width=True, hide_index=True,
)

fig_indiv = px.bar(indiv_df, x="Model", y="Test Accuracy", color="Model",
                   color_discrete_sequence=["#E63946", "#2A9D8F", "#264653", "#FB8500", "#7209B7"],
                   title="Individual Base Model Test Accuracy")
apply_common_layout(fig_indiv, height=400)
fig_indiv.update_layout(xaxis_tickangle=-20)
st.plotly_chart(fig_indiv, use_container_width=True)

# ── 49.4 Interactive Model Selection for Stacking ────────────────────────────
st.header("49.3  Interactive: Build Your Stacking Ensemble")

st.markdown(
    "Now for the fun part: pick your team. Select which base models to include in "
    "the stacking ensemble and see if the whole is greater than the sum of its parts."
)

selected_models = st.multiselect(
    "Base Learners",
    options=list(available_models.keys()),
    default=["Logistic Regression", "Random Forest", "Gradient Boosting"],
    key="stack_models",
)

meta_learner_choice = st.selectbox(
    "Meta-Learner",
    ["Logistic Regression", "Random Forest (shallow)"],
    key="meta_learner",
)

if len(selected_models) < 2:
    st.warning("Please select at least 2 base models for stacking. A one-person team is not really a team.")
else:
    # Build estimators list
    estimators = [(name, available_models[name]) for name in selected_models]

    # Meta-learner
    if meta_learner_choice == "Logistic Regression":
        meta = LogisticRegression(max_iter=1000, random_state=42)
    else:
        meta = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)

    # Stacking
    with st.spinner("Training stacking ensemble..."):
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=meta,
            cv=5,
            n_jobs=-1,
        )
        stack.fit(X_tr_s, y_tr_s)
        stack_train_acc = accuracy_score(y_tr_s, stack.predict(X_tr_s))
        stack_test_acc = accuracy_score(y_test, stack.predict(X_test))

    # Also compare with simple voting
    voting = VotingClassifier(estimators=estimators, voting="hard", n_jobs=-1)
    voting.fit(X_tr_s, y_tr_s)
    voting_test_acc = accuracy_score(y_test, voting.predict(X_test))

    st.subheader("Results")

    col1, col2, col3 = st.columns(3)
    best_individual = indiv_df[indiv_df["Model"].isin(selected_models)]["Test Accuracy"].max()
    col1.metric("Best Individual Base", f"{best_individual:.4f}")
    col2.metric("Hard Voting Ensemble", f"{voting_test_acc:.4f}",
                delta=f"{voting_test_acc - best_individual:+.4f}")
    col3.metric("Stacking Ensemble", f"{stack_test_acc:.4f}",
                delta=f"{stack_test_acc - best_individual:+.4f}")

    # Comprehensive comparison
    comp_data = []
    for name in selected_models:
        row = indiv_df[indiv_df["Model"] == name].iloc[0]
        comp_data.append({"Model": name, "Test Accuracy": row["Test Accuracy"], "Type": "Base Learner"})
    comp_data.append({"Model": "Hard Voting", "Test Accuracy": voting_test_acc, "Type": "Voting Ensemble"})
    comp_data.append({"Model": "Stacking", "Test Accuracy": stack_test_acc, "Type": "Stacking Ensemble"})

    comp_df = pd.DataFrame(comp_data).sort_values("Test Accuracy", ascending=False)

    fig_comp = px.bar(comp_df, x="Model", y="Test Accuracy", color="Type",
                      color_discrete_map={
                          "Base Learner": "#264653",
                          "Voting Ensemble": "#2A9D8F",
                          "Stacking Ensemble": "#E63946",
                      },
                      title="Base Learners vs Voting vs Stacking")
    apply_common_layout(fig_comp, height=450)
    fig_comp.update_layout(xaxis_tickangle=-20)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.dataframe(
        comp_df.style.format({"Test Accuracy": "{:.4f}"})
        .highlight_max(subset=["Test Accuracy"], color="#d4edda"),
        use_container_width=True, hide_index=True,
    )

    if stack_test_acc > best_individual:
        st.success(
            f"Stacking improved over the best base learner by "
            f"**{(stack_test_acc - best_individual)*100:.2f} percentage points**! "
            "The meta-learner found value in combining the models."
        )
    else:
        st.info(
            "Stacking did not improve over the best base learner here. "
            "This sometimes happens when one model is already so good that the others "
            "just add noise, or when the base models are not diverse enough -- they all "
            "make the same mistakes, so there is nothing for the meta-learner to learn."
        )

    # Confusion matrix for stacking
    st.subheader("Stacking Confusion Matrix")
    y_stack_pred = stack.predict(X_test)
    metrics_stack = classification_metrics(y_test, y_stack_pred, labels=city_labels)
    fig_cm = plot_confusion_matrix(metrics_stack["confusion_matrix"], city_labels)
    st.plotly_chart(fig_cm, use_container_width=True)

# ── 49.5 Model Diversity Analysis ────────────────────────────────────────────
st.header("49.4  Why Model Diversity Matters")

concept_box(
    "Diversity is Key",
    "Here is the thing about stacking that is not immediately obvious: it only works "
    "when the base models make <b>different errors</b>. If every model is confused by "
    "the same examples, combining them accomplishes nothing -- you are just doing the "
    "same wrong thing five times. The magic happens when model A gets example 17 wrong "
    "but model B gets it right, and the meta-learner figures out to trust B on examples "
    "like that. This is why you want models with <b>different inductive biases</b>: "
    "linear models, tree models, distance-based models -- they see the data differently."
)

# Show prediction agreement matrix
st.subheader("Prediction Agreement Between Base Models")
predictions = {}
for name, model in available_models.items():
    model.fit(X_tr_s, y_tr_s)
    predictions[name] = model.predict(X_test)

agreement_matrix = pd.DataFrame(index=available_models.keys(), columns=available_models.keys(), dtype=float)
for m1 in available_models:
    for m2 in available_models:
        agreement_matrix.loc[m1, m2] = (predictions[m1] == predictions[m2]).mean()

fig_agree = go.Figure(data=go.Heatmap(
    z=agreement_matrix.values.astype(float),
    x=list(agreement_matrix.columns),
    y=list(agreement_matrix.index),
    colorscale="RdYlGn",
    text=np.round(agreement_matrix.values.astype(float), 3),
    texttemplate="%{text}",
))
apply_common_layout(fig_agree, title="Prediction Agreement Matrix", height=450)
fig_agree.update_layout(xaxis_title="Model", yaxis_title="Model")
st.plotly_chart(fig_agree, use_container_width=True)

insight_box(
    "The most useful pairs for stacking are those with agreement around 0.7-0.8. "
    "They agree on the easy cases (which is reassuring) but disagree on the hard "
    "ones (which is where the meta-learner can add value). If two models agree 95%+ "
    "of the time, one of them is redundant. If they agree less than 60%, at least one "
    "of them might just be bad."
)

# ── 49.6 Stacking vs Voting vs Bagging vs Boosting ──────────────────────────
st.header("49.5  Ensemble Methods Summary")

summary_df = pd.DataFrame({
    "Method": ["Bagging", "Boosting", "Voting", "Stacking"],
    "Strategy": [
        "Train same model on bootstrap samples",
        "Train models sequentially on residuals",
        "Take majority vote / average",
        "Train meta-learner on base predictions",
    ],
    "Diversity Source": [
        "Random samples + random features",
        "Focus on different errors each round",
        "Different model types",
        "Different model types + learned combination",
    ],
    "Reduces": ["Variance", "Bias", "Both (somewhat)", "Both"],
    "Complexity": ["Low", "Medium", "Low", "High"],
})
st.dataframe(summary_df, use_container_width=True, hide_index=True)

code_example("""from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Define base learners
estimators = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=50)),
    ('gb', GradientBoostingClassifier(n_estimators=50)),
]

# Simple voting
voting = VotingClassifier(estimators=estimators, voting='hard')
voting.fit(X_train, y_train)

# Stacking with meta-learner
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
)
stacking.fit(X_train, y_train)
print(f"Stacking accuracy: {stacking.score(X_test, y_test):.4f}")
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "What role does the meta-learner play in stacking?",
    [
        "It replaces all base models",
        "It learns the optimal way to combine base model predictions",
        "It trains the base models faster",
        "It performs feature selection",
    ],
    correct_idx=1,
    explanation="The meta-learner takes base model predictions as its input features and learns how to combine them optimally. It is the manager who figures out which specialist to listen to in which situation.",
    key="ch49_quiz1",
)

quiz(
    "Stacking works best when base models are:",
    [
        "All the same algorithm with different hyperparameters",
        "As diverse as possible (different algorithms, different errors)",
        "Trained on the same features only",
        "All very complex models",
    ],
    correct_idx=1,
    explanation="Diversity is the key ingredient. If all your models make the same mistakes, combining them cannot help. You want models that are individually decent but wrong in different ways.",
    key="ch49_quiz2",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "**Stacking** uses a meta-learner to combine diverse base model predictions -- not just voting, but learned, optimized combination.",
    "Base models should be as **diverse** as possible. Mix linear models, tree models, and distance-based models for maximum complementarity.",
    "The meta-learner is typically a simple model (logistic regression) to avoid overfitting. You do not want the manager to be more complicated than the team.",
    "Cross-validation is used internally to generate honest out-of-fold predictions for training the meta-learner, avoiding leakage.",
    "Stacking can outperform individual models and simple voting, but not always -- it depends on whether the base models bring genuinely different perspectives.",
    "Use the **prediction agreement matrix** to assess diversity: look for 0.7-0.8 agreement between pairs.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 48: Boosting",
    prev_page="48_Boosting.py",
    next_label="Ch 50: Neural Network Basics",
    next_page="50_Neural_Network_Basics.py",
)
