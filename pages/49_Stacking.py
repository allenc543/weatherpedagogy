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

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: What are we doing and why
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "Let me remind you of the exact task we've been working on, because everything in "
    "this chapter builds on it."
)
st.markdown(
    "**The task**: We have ~105,000 hourly weather readings from 6 US cities (Dallas, "
    "San Antonio, Houston, Austin, New York, Los Angeles). Each reading is a row with "
    "4 numbers: temperature in Celsius, relative humidity as a percentage, wind speed "
    "in km/h, and surface pressure in hPa. Given *just those 4 numbers* -- no timestamps, "
    "no city labels, nothing else -- we're trying to predict which city the reading came "
    "from."
)
st.markdown(
    "Over the past few chapters, we've thrown different algorithms at this problem. "
    "Logistic regression draws linear decision boundaries in 4D space. Random forests "
    "ask sequences of yes/no questions ('Is temperature above 22°C? If yes, is humidity "
    "below 50%?'). KNN looks at the 7 most similar readings in the training data and "
    "goes with the majority city. Gradient boosting builds trees that each focus on the "
    "mistakes of previous trees."
)
st.markdown(
    "Each of these models gets a different accuracy. None of them is perfect. But here's "
    "the interesting observation: **they're wrong about different things**. The random "
    "forest might confuse Dallas and San Antonio (similar Texas heat) but correctly "
    "identify NYC. The logistic regression might nail the Dallas/San Antonio distinction "
    "(slightly different humidity profiles) but struggle with Houston vs Austin. "
    "What if we could combine their strengths?"
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: The idea, built up from scratch
# ─────────────────────────────────────────────────────────────────────────────
st.header("From Voting to Stacking: Building the Intuition")

st.markdown(
    "The simplest combination strategy is **voting**: ask all 5 models 'which city?', "
    "take the majority answer. If 3 out of 5 say Houston, the final answer is Houston. "
    "This is already pretty good -- it's the wisdom of crowds applied to algorithms."
)
st.markdown(
    "But voting treats every model equally, and that's not always fair. Suppose the "
    "gradient boosting model gets 65% accuracy and the decision tree gets 48%. Should "
    "their votes count the same? Probably not. What if the KNN model is great at "
    "identifying LA (dry, warm, low wind) but terrible at distinguishing the Texas "
    "cities? You'd want to trust KNN when it says 'LA' but be skeptical when it says "
    "'Dallas.'"
)
st.markdown(
    "**Stacking** takes this idea all the way. Instead of simple voting, you train a "
    "*second model* -- called the **meta-learner** -- whose job is to look at all the "
    "base models' predictions and figure out the optimal way to combine them. The "
    "meta-learner's input isn't weather data. Its input is: 'Model 1 said Houston, "
    "Model 2 said Dallas, Model 3 said Houston, Model 4 said San Antonio, Model 5 "
    "said Houston.' From those 5 predictions, the meta-learner outputs a final answer."
)
st.markdown(
    "Think of it as a two-layer system. The bottom layer has 5 specialists who each "
    "look at the raw weather reading and make their best guess. The top layer has one "
    "manager who doesn't look at the weather data at all -- the manager only sees the "
    "specialists' guesses and decides whose opinion to trust."
)

concept_box(
    "The Two Levels, Concretely",
    "Say a weather reading comes in: 28.3°C, 72% humidity, 8.4 km/h wind, 1015.2 hPa.<br><br>"
    "<b>Level 0 (Base learners)</b> each get this reading and produce a prediction:<br>"
    "- Logistic Regression → Houston<br>"
    "- Random Forest → Houston<br>"
    "- Gradient Boosting → San Antonio<br>"
    "- KNN → Dallas<br>"
    "- Decision Tree → Houston<br><br>"
    "<b>Level 1 (Meta-learner)</b> sees [Houston, Houston, San Antonio, Dallas, Houston] "
    "and outputs: <b>Houston</b> (confidence: 78%).<br><br>"
    "Simple voting would also say Houston (3 out of 5). But the meta-learner does "
    "something smarter: it has <em>learned</em> that when Gradient Boosting disagrees "
    "with the majority on Texas cities, Gradient Boosting is wrong about 60% of the time. "
    "So it downweights that dissent. And it has learned that KNN's Dallas predictions "
    "are unreliable when humidity is high (which it can infer from the pattern of other "
    "models' predictions). Voting can't do any of this.",
)

formula_box(
    "Stacking Prediction",
    r"\underbrace{\hat{y}}_{\text{final prediction}} = \underbrace{g}_{\text{meta-learner}}\!\left(\underbrace{\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_M}_{\text{base model predictions}}\right)",
    "g() is the meta-learner; y_hat_1..M are the base model predictions. The meta-learner "
    "takes M predictions as input features and outputs the final prediction. It's just "
    "another classifier, but trained on predictions instead of weather data."
)

warning_box(
    "There's a subtle trap here. When generating predictions for the meta-learner to "
    "train on, the base models MUST make predictions on data they weren't trained on. "
    "Otherwise you'd be feeding the meta-learner predictions that are unrealistically "
    "good (the base models have seen those examples before), and the meta-learner "
    "would learn to trust the base models more than it should. The solution: use "
    "cross-validation internally. Each base model predicts on its held-out fold, so "
    "the meta-learner only ever sees honest, out-of-sample predictions."
)

st.divider()

# ── Prepare Data ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Individual base model performance
# ─────────────────────────────────────────────────────────────────────────────
st.header("First: How Does Each Model Do Alone?")

st.markdown(
    "Before we combine anything, let's see what each individual model achieves on our "
    "city-from-weather task. We train each model on 5,000 weather readings (subsampled "
    "for speed) and test on a held-out 20% of the full dataset. The 4 input features "
    "are always the same: temperature, humidity, wind speed, and pressure. The output "
    "is always one of 6 city labels."
)

available_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
}

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

st.markdown(
    "**How to read this table**: Train accuracy is how well the model does on data it "
    "was trained on (easy -- it's seen the answers). Test accuracy is what matters: how "
    "well does it predict cities from weather readings it has *never seen before*? CV Mean "
    "is the average 5-fold cross-validation accuracy, a more robust estimate."
)

st.dataframe(
    indiv_df.style.format({
        "Train Accuracy": "{:.4f}", "Test Accuracy": "{:.4f}",
        "CV Mean": "{:.4f}", "CV Std": "{:.4f}",
    }).highlight_max(subset=["Test Accuracy"], color="#d4edda"),
    use_container_width=True, hide_index=True,
)

fig_indiv = px.bar(indiv_df, x="Model", y="Test Accuracy", color="Model",
                   color_discrete_sequence=["#E63946", "#2A9D8F", "#264653", "#FB8500", "#7209B7"],
                   title="Individual Model Test Accuracy: Predicting City from 4 Weather Features")
apply_common_layout(fig_indiv, height=400)
fig_indiv.update_layout(xaxis_tickangle=-20)
st.plotly_chart(fig_indiv, use_container_width=True)

best_model_name = indiv_df.iloc[0]["Model"]
best_model_acc = indiv_df.iloc[0]["Test Accuracy"]
worst_model_name = indiv_df.iloc[-1]["Model"]
worst_model_acc = indiv_df.iloc[-1]["Test Accuracy"]

st.markdown(
    f"The best individual model is **{best_model_name}** at {best_model_acc:.1%} test "
    f"accuracy. The worst is **{worst_model_name}** at {worst_model_acc:.1%}. Remember: "
    f"random guessing among 6 cities would give 16.7%. So even the worst model is "
    f"learning something real about how cities differ in their weather fingerprints. "
    f"The question is: can we do better by combining them?"
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Interactive stacking
# ─────────────────────────────────────────────────────────────────────────────
st.header("Build Your Own Stacking Ensemble")

st.markdown(
    "Pick which models go on your team (the base learners) and which model acts as "
    "the manager (the meta-learner). Then we'll compare the stacking ensemble against "
    "the best individual model and against simple majority voting."
)

selected_models = st.multiselect(
    "Base Learners (pick at least 2)",
    options=list(available_models.keys()),
    default=["Logistic Regression", "Random Forest", "Gradient Boosting"],
    key="stack_models",
)

meta_learner_choice = st.selectbox(
    "Meta-Learner (the 'manager' model)",
    ["Logistic Regression", "Random Forest (shallow)"],
    key="meta_learner",
    help="The meta-learner should be simple. Its input is just the base models' predictions "
         "(a handful of numbers), not the raw weather data. A complex meta-learner will overfit."
)

if len(selected_models) < 2:
    st.warning("Select at least 2 base models. Stacking with one model is just... using that model.")
else:
    # Build estimators list
    estimators = [(name, available_models[name]) for name in selected_models]

    # Meta-learner
    if meta_learner_choice == "Logistic Regression":
        meta = LogisticRegression(max_iter=1000, random_state=42)
    else:
        meta = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)

    # Stacking
    with st.spinner("Training stacking ensemble (5-fold CV internally to generate meta-learner training data)..."):
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

    st.subheader("The Verdict")

    col1, col2, col3 = st.columns(3)
    best_individual = indiv_df[indiv_df["Model"].isin(selected_models)]["Test Accuracy"].max()
    col1.metric("Best Individual Base", f"{best_individual:.4f}",
                help="The best single model among your selected base learners")
    col2.metric("Hard Voting Ensemble", f"{voting_test_acc:.4f}",
                delta=f"{voting_test_acc - best_individual:+.4f}",
                help="Simple majority vote: each model gets one equal vote")
    col3.metric("Stacking Ensemble", f"{stack_test_acc:.4f}",
                delta=f"{stack_test_acc - best_individual:+.4f}",
                help="Meta-learner decides how to weight each model's prediction")

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
                      title="City Prediction Accuracy: Individual Models vs Ensembles")
    apply_common_layout(fig_comp, height=450)
    fig_comp.update_layout(xaxis_tickangle=-20)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.dataframe(
        comp_df.style.format({"Test Accuracy": "{:.4f}"})
        .highlight_max(subset=["Test Accuracy"], color="#d4edda"),
        use_container_width=True, hide_index=True,
    )

    if stack_test_acc > best_individual:
        improvement = (stack_test_acc - best_individual) * 100
        st.success(
            f"Stacking improved over the best individual model by "
            f"**{improvement:.2f} percentage points**. That might sound small, "
            f"but in a 6-class problem where the best model is already decent, "
            f"squeezing out extra accuracy is hard. The meta-learner found patterns "
            f"in when to trust which model."
        )
    elif stack_test_acc == best_individual:
        st.info(
            "Stacking matched but didn't beat the best individual model. This happens "
            "when one model is already dominant -- it gets most things right, so the "
            "meta-learner essentially learns to just defer to it."
        )
    else:
        st.info(
            "Stacking didn't beat the best individual model here. This can happen when: "
            "(1) the base models aren't diverse enough -- they all confuse the same "
            "city pairs, so combining them doesn't help; (2) the meta-learner overfits "
            "to the cross-validation predictions; or (3) the best single model is already "
            "so good there's little room for improvement."
        )

    # Confusion matrix for stacking
    st.subheader("Where Does the Stacking Ensemble Still Struggle?")
    st.markdown(
        "The confusion matrix shows which cities the stacking ensemble confuses. "
        "Read it row-by-row: each row is the true city, each column is the predicted "
        "city. Bright diagonal = correct predictions. Off-diagonal = errors."
    )
    y_stack_pred = stack.predict(X_test)
    metrics_stack = classification_metrics(y_test, y_stack_pred, labels=city_labels)
    fig_cm = plot_confusion_matrix(metrics_stack["confusion_matrix"], city_labels)
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown(
        "Look at the off-diagonal cells. The biggest numbers there show which city "
        "pairs are hardest to tell apart from weather alone. Dallas and San Antonio? "
        "Houston and Austin? These are cities with genuinely similar climates, so even "
        "the combined wisdom of 5 different algorithms struggles to distinguish them "
        "from a single weather reading."
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Model diversity
# ─────────────────────────────────────────────────────────────────────────────
st.header("Why Stacking Works (When It Works): Model Diversity")

st.markdown(
    "Here's the key insight: stacking only helps if the base models **disagree about "
    "different things**. If every model confuses Dallas with San Antonio in the exact "
    "same situations, combining them is like asking 5 people with the same blindness "
    "to help you see. You need models with *different* blind spots."
)
st.markdown(
    "Why would different algorithms disagree? Because they work fundamentally "
    "differently:\n\n"
    "- **Logistic Regression** draws straight-line boundaries in 4D weather-feature "
    "space. It might find that 'humidity above 70% AND temperature above 25°C → Houston' "
    "but can't capture curved boundaries.\n"
    "- **Random Forest** asks sequential yes/no questions and can capture complex "
    "interactions ('high temperature + low wind + moderate humidity → Austin') but "
    "might miss smooth gradients between cities.\n"
    "- **KNN** says 'find the 7 most similar weather readings in the training data "
    "and go with their majority city.' It's great at capturing local patterns but "
    "sensitive to feature scaling.\n\n"
    "These different approaches mean they'll be right about *different* subsets of "
    "weather readings, which is exactly what the meta-learner exploits."
)

# Show prediction agreement matrix
st.subheader("How Often Do the Models Agree?")
st.markdown(
    "This heatmap shows, for each pair of models, what fraction of test readings they "
    "predict the same city for. 1.0 = perfect agreement (they always say the same thing). "
    "0.17 = they might as well be independent random guessers."
)

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
    "The sweet spot for stacking is agreement around 0.7-0.8 between model pairs. "
    "That means they agree on the easy readings (a 35°C, 25% humidity reading is "
    "obviously not NYC in winter -- every model knows that) but disagree on the hard "
    "ones (a 22°C, 60% humidity reading could plausibly be several cities). If two "
    "models agree 95%+ of the time, one of them is redundant -- the meta-learner "
    "can't learn anything by having both."
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Summary comparison
# ─────────────────────────────────────────────────────────────────────────────
st.header("All the Ensemble Methods We've Learned, Side by Side")

st.markdown(
    "We've now covered four ensemble strategies across three chapters. Here's how "
    "they compare -- all applied to the same task of predicting city from weather:"
)

summary_df = pd.DataFrame({
    "Method": ["Bagging (Ch 47)", "Boosting (Ch 48)", "Voting", "Stacking"],
    "Strategy": [
        "Train same model (e.g. decision tree) on random subsets of the data",
        "Train models sequentially, each one focusing on the previous one's errors",
        "Train different models, take majority vote",
        "Train different models, then train a meta-learner to combine their predictions",
    ],
    "What it combines": [
        "Same algorithm, different data samples",
        "Same algorithm, different error focus",
        "Different algorithms, equal weight",
        "Different algorithms, learned weights",
    ],
    "Main benefit": [
        "Reduces variance (less overfitting)",
        "Reduces bias (better accuracy)",
        "Simple error cancellation",
        "Optimal combination (when models are diverse)",
    ],
})
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.divider()

code_example("""from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Our task: predict city from 4 weather features
# X = [[temp, humidity, wind, pressure], ...] — shape (n_samples, 4)
# y = ['Dallas', 'NYC', 'Houston', ...] — city labels

# Define base learners (diverse algorithms)
estimators = [
    ('lr', LogisticRegression(max_iter=1000)),    # linear boundaries
    ('rf', RandomForestClassifier(n_estimators=50)),  # tree-based
    ('gb', GradientBoostingClassifier(n_estimators=50)),  # boosted trees
]

# Simple voting: each model gets one vote, majority wins
voting = VotingClassifier(estimators=estimators, voting='hard')
voting.fit(X_train, y_train)

# Stacking: meta-learner (logistic regression) learns to combine predictions
# cv=5 means base models use 5-fold CV to generate meta-learner training data
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
    "A weather reading is 28°C, 71% humidity, 9 km/h wind, 1014 hPa. The logistic "
    "regression says Houston, the random forest says Houston, the KNN says Dallas, "
    "gradient boosting says San Antonio, and the decision tree says Houston. "
    "What does a simple majority vote predict?",
    [
        "Dallas",
        "San Antonio",
        "Houston (3 out of 5 votes)",
        "It's a tie",
    ],
    correct_idx=2,
    explanation=(
        "Three models say Houston, one says Dallas, one says San Antonio. Majority "
        "vote goes with Houston. Stacking might reach the same conclusion but for a "
        "different reason: it has learned that when the logistic regression and random "
        "forest agree on a Texas city, they're usually right -- so it weighs their "
        "agreement more heavily than the raw vote count."
    ),
    key="ch49_quiz1",
)

quiz(
    "Your 5 base models agree on the predicted city 95% of the time. Will stacking help much?",
    [
        "Yes — more models always helps",
        "No — if they always agree, combining them adds nothing new",
        "Only if the meta-learner is very complex",
        "Only if you add more features",
    ],
    correct_idx=1,
    explanation=(
        "Stacking exploits *disagreement* between models. If they agree 95% of the time, "
        "the meta-learner can only do something useful on the remaining 5% of readings. "
        "And even on those, it might not help much. You'd get more from adding a genuinely "
        "different type of model (say, an SVM if you only have tree-based models) to create "
        "more diversity."
    ),
    key="ch49_quiz2",
)

quiz(
    "Why must base model predictions for the meta-learner be generated via cross-validation?",
    [
        "To make training faster",
        "To prevent data leakage — the meta-learner would see unrealistically good predictions otherwise",
        "To increase model diversity",
        "Because sklearn requires it",
    ],
    correct_idx=1,
    explanation=(
        "If you let base models predict on data they trained on, their predictions would "
        "be unrealistically accurate (they've seen the answers). The meta-learner would "
        "learn to trust them unconditionally. Then at test time, when base model "
        "predictions are honest (and worse), the meta-learner wouldn't be calibrated "
        "for that. Cross-validation ensures the meta-learner only sees predictions of "
        "the same quality it will encounter at test time."
    ),
    key="ch49_quiz3",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "**The task**: predict which of 6 cities a weather reading came from, using 4 "
    "features (temperature, humidity, wind speed, pressure). Stacking combines "
    "multiple models' predictions via a learned meta-model.",
    "**Stacking vs voting**: Voting treats all models equally. Stacking learns which "
    "model to trust in which situation -- it might trust KNN when it says 'LA' but "
    "defer to gradient boosting for Texas cities.",
    "**Diversity is everything**: Stacking only helps when base models make different "
    "errors. Five copies of the same algorithm give you nothing. Mix linear, tree-based, "
    "and distance-based models.",
    "**The meta-learner should be simple** (usually logistic regression). Its input is "
    "just a handful of predictions, not the full weather data. A complex meta-learner "
    "overfits to noise in the base models' cross-validation predictions.",
    "**Cross-validation prevents leakage**: Base model predictions for meta-learner "
    "training must be out-of-fold, so the meta-learner sees honest predictions.",
    "**Check the agreement matrix**: 0.7-0.8 agreement between model pairs is ideal. "
    "Too high = redundancy. Too low = at least one model is bad.",
])

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
navigation(
    prev_label="Ch 48: Boosting",
    prev_page="48_Boosting.py",
    next_label="Ch 50: Neural Network Basics",
    next_page="50_Neural_Network_Basics.py",
)
