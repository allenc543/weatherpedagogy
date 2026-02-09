"""Chapter 42 -- Feature Importance: Permutation, MDI, and SHAP."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

from utils.data_loader import load_data, sidebar_filters
from utils.plotting import apply_common_layout
from utils.constants import CITY_COLORS, CITY_LIST, FEATURE_COLS, FEATURE_LABELS
from utils.ui_components import (
    chapter_header, concept_box, formula_box, insight_box,
    warning_box, code_example, quiz, takeaways,
)

# ── Page config ──────────────────────────────────────────────────────────────
chapter_header(42, "Feature Importance", part="IX")

st.markdown(
    "You have trained a model. It works. Congratulations. Now comes the question "
    "that separates data scientists from fortune tellers: **why** does it work? "
    "Which features does it actually rely on, and how do they influence its "
    "predictions? Feature importance methods answer this, and it turns out there "
    "are at least three different ways to ask the question, each with its own "
    "philosophy and its own failure modes. We are going to try all three and see "
    "where they agree (which is where you should be confident) and where they "
    "disagree (which is where you should be curious)."
)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
fdf = sidebar_filters(df)

concept_box(
    "Three Ways to Measure Feature Importance",
    "<b>MDI (Mean Decrease in Impurity)</b>: Built into tree models. For every "
    "split in every tree, it measures how much that split reduced impurity (Gini "
    "or entropy). Sum it up across all trees and you get importance. It is fast "
    "and free (the model already computed it), but it has a known bias toward "
    "high-cardinality and correlated features.<br>"
    "<b>Permutation Importance</b>: The conceptually cleanest method. Shuffle one "
    "feature at a time in the test set and measure how much accuracy drops. If "
    "shuffling temperature destroys the model, temperature was important. It is "
    "model-agnostic and unbiased, but slow.<br>"
    "<b>SHAP (SHapley Additive exPlanations)</b>: The nuclear option. Based on "
    "Shapley values from game theory, SHAP computes each feature's marginal "
    "contribution to every individual prediction. It gives you global importance "
    "*and* explains individual predictions. The gold standard, but computationally "
    "expensive."
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.subheader("Importance Settings")
model_choice = st.sidebar.selectbox(
    "Model",
    ["Random Forest", "Gradient Boosting (XGBoost-style)"],
    key="fi_model",
)

# ── Prepare data ─────────────────────────────────────────────────────────────
# Build extended features
sub = fdf.dropna(subset=FEATURE_COLS).sample(n=min(8000, len(fdf)), random_state=42).copy()
sub["hour_sin"] = np.sin(2 * np.pi * sub["hour"] / 24)
sub["hour_cos"] = np.cos(2 * np.pi * sub["hour"] / 24)
sub["month_sin"] = np.sin(2 * np.pi * sub["month"] / 12)
sub["month_cos"] = np.cos(2 * np.pi * sub["month"] / 12)
sub["dew_point"] = sub["temperature_c"] - (100 - sub["relative_humidity_pct"]) / 5

all_features = FEATURE_COLS + [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dew_point",
]
feature_display = {
    **FEATURE_LABELS,
    "hour_sin": "Hour (sin)", "hour_cos": "Hour (cos)",
    "month_sin": "Month (sin)", "month_cos": "Month (cos)",
    "dew_point": "Dew Point (est.)",
}

le = LabelEncoder()
X = sub[all_features].values
y = le.fit_transform(sub["city"])
city_labels = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

# Train model
if model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
else:
    model = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)

model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

st.metric(f"{model_choice} Accuracy (city classification)", f"{acc:.1%}")

# ── Section 1: MDI (tree-based importance) ───────────────────────────────────
st.header("1. Mean Decrease in Impurity (MDI)")

st.markdown(
    "MDI is the feature importance you get for free with any tree-based model. "
    "Every time a tree makes a split on, say, humidity, it reduces the impurity "
    "(mess) in the resulting child nodes. MDI adds up all those reductions across "
    "all trees and normalizes. It is fast, it is convenient, and it is a reasonable "
    "first approximation -- but it has known blind spots."
)

mdi = model.feature_importances_
mdi_df = pd.DataFrame({
    "Feature": [feature_display.get(f, f) for f in all_features],
    "Importance": mdi,
}).sort_values("Importance", ascending=True)

fig_mdi = go.Figure()
fig_mdi.add_trace(go.Bar(
    x=mdi_df["Importance"], y=mdi_df["Feature"],
    orientation="h", marker_color="#2E86C1",
))
fig_mdi.update_layout(xaxis_title="MDI Importance")
apply_common_layout(fig_mdi, f"MDI Feature Importance ({model_choice})", 500)
st.plotly_chart(fig_mdi, use_container_width=True)

warning_box(
    "MDI has a well-documented bias: it inflates the importance of features with "
    "many unique values (high cardinality) and features that are correlated with "
    "each other. If two features are highly correlated, the model will split on "
    "whichever one it encounters first, making that one look important and the "
    "other look useless -- even though they carry the same information. Permutation "
    "importance does not have this problem."
)

# ── Section 2: Permutation Importance ────────────────────────────────────────
st.header("2. Permutation Importance")

st.markdown(
    "Permutation importance is the conceptual equivalent of asking: 'What happens "
    "if I break this feature?' For each feature, we randomly shuffle its values in "
    "the test set (destroying its relationship with the target) and measure how "
    "much accuracy drops. A big drop means the model was relying on that feature. "
    "No drop means it was irrelevant -- the model can do just fine without it."
)

with st.spinner("Computing permutation importance (10 repeats)..."):
    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1,
    )

perm_df = pd.DataFrame({
    "Feature": [feature_display.get(f, f) for f in all_features],
    "Mean Importance": perm_result.importances_mean,
    "Std": perm_result.importances_std,
}).sort_values("Mean Importance", ascending=True)

fig_perm = go.Figure()
fig_perm.add_trace(go.Bar(
    x=perm_df["Mean Importance"], y=perm_df["Feature"],
    orientation="h", marker_color="#E63946",
    error_x=dict(type="data", array=perm_df["Std"].values),
))
fig_perm.update_layout(xaxis_title="Accuracy Drop When Shuffled")
apply_common_layout(fig_perm, "Permutation Feature Importance", 500)
st.plotly_chart(fig_perm, use_container_width=True)

# Compare MDI vs Permutation
st.subheader("MDI vs Permutation: Do They Agree?")

compare_df = pd.DataFrame({
    "Feature": [feature_display.get(f, f) for f in all_features],
    "MDI Rank": mdi_df.sort_values("Importance", ascending=False).reset_index(drop=True).index + 1,
    "Permutation Rank": perm_df.sort_values("Mean Importance", ascending=False).reset_index(drop=True).index + 1,
})
# Re-sort by feature name for alignment
mdi_ranked = mdi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
mdi_ranked["MDI Rank"] = range(1, len(mdi_ranked) + 1)
perm_ranked = perm_df.sort_values("Mean Importance", ascending=False).reset_index(drop=True)
perm_ranked["Perm Rank"] = range(1, len(perm_ranked) + 1)

combined = mdi_ranked[["Feature", "MDI Rank"]].merge(
    perm_ranked[["Feature", "Perm Rank"]], on="Feature",
).sort_values("MDI Rank")
st.dataframe(combined, use_container_width=True, hide_index=True)

insight_box(
    "MDI and permutation importance usually agree on the top 2-3 features, but "
    "they can diverge significantly on the rest. When they disagree, permutation "
    "importance is generally the more trustworthy one, because it directly measures "
    "the impact on prediction accuracy rather than relying on a proxy (impurity "
    "reduction). Think of MDI as the quick estimate and permutation as the careful "
    "measurement."
)

# ── Section 3: SHAP Values ──────────────────────────────────────────────────
st.header("3. SHAP Values")

concept_box(
    "What Are SHAP Values?",
    "SHAP values come from a place you would not expect: cooperative game theory. "
    "Imagine each feature is a 'player' on a team, and the 'game' is making a "
    "prediction. The Shapley value asks: what is each player's fair share of the "
    "credit? It computes this by considering every possible coalition of features "
    "and averaging each feature's marginal contribution across all of them. "
    "The result is locally accurate (the SHAP values for one prediction sum "
    "to the model output), consistent, and deeply principled. It is the closest "
    "thing we have to a 'why did the model say this?' button."
)

formula_box(
    "Shapley Value",
    r"\underbrace{\phi_i}_{\text{feature i credit}} = \sum_{\underbrace{S \subseteq F \setminus \{i\}}_{\text{all feature subsets}}} "
    r"\underbrace{\frac{|S|!\,(|F|-|S|-1)!}{|F|!}}_{\text{fairness weight}} "
    r"\left[ \underbrace{f(S \cup \{i\})}_{\text{with feature i}} - \underbrace{f(S)}_{\text{without feature i}} \right]",
    "The weighted average of feature i's marginal contribution across all "
    "possible coalitions S of other features. The combinatorial weighting "
    "ensures fairness -- features that contribute more in more contexts get "
    "more credit."
)

try:
    import shap

    st.markdown(
        "We compute SHAP values for a subsample of test observations. "
        "The **summary plot** below shows both *which* features are important "
        "and *how* they affect predictions -- something neither MDI nor "
        "permutation importance can tell you."
    )

    # Use a subsample for SHAP speed
    shap_sample_size = min(500, len(X_test))
    X_shap = X_test[:shap_sample_size]
    y_shap = y_test[:shap_sample_size]

    with st.spinner("Computing SHAP values..."):
        if model_choice == "Random Forest":
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

    shap_available = True

except ImportError:
    st.info(
        "Install the `shap` package (`pip install shap`) for SHAP analysis. "
        "We will show a simulation of SHAP-like insights instead."
    )
    shap_available = False

if shap_available:
    # SHAP has shape (n_classes, n_samples, n_features) for multiclass
    # or (n_samples, n_features) for binary
    if isinstance(shap_values, list):
        # Multi-class: average absolute SHAP across classes
        mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values)

    # Global importance from SHAP
    shap_importance = np.mean(mean_abs_shap, axis=0)
    shap_df = pd.DataFrame({
        "Feature": [feature_display.get(f, f) for f in all_features],
        "Mean |SHAP|": shap_importance,
    }).sort_values("Mean |SHAP|", ascending=True)

    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        x=shap_df["Mean |SHAP|"], y=shap_df["Feature"],
        orientation="h", marker_color="#2A9D8F",
    ))
    fig_shap.update_layout(xaxis_title="Mean |SHAP Value|")
    apply_common_layout(fig_shap, "SHAP Feature Importance (Global)", 500)
    st.plotly_chart(fig_shap, use_container_width=True)

    # ── SHAP Summary-like plot (beeswarm approximation) ──────────────────────
    st.subheader("SHAP Summary: Feature Value vs SHAP Impact")

    st.markdown(
        "This is the plot that makes SHAP worth the computational cost. Each dot "
        "is one observation. The x-axis shows the SHAP value (positive = pushes "
        "prediction toward this class, negative = pushes away). The color shows "
        "the feature value (red = high, blue = low). So if you see 'high humidity "
        "(red dots) clustered on the right,' it means high humidity pushes predictions "
        "toward this city. This is *directional* insight that no other method provides."
    )

    # Pick the class with the most interesting pattern or the first class
    # For demonstration, show the SHAP values averaged across classes
    if isinstance(shap_values, list) and len(shap_values) > 0:
        # Show for the Houston class (typically class index for Houston)
        houston_idx = np.where(city_labels == "Houston")[0]
        la_idx = np.where(city_labels == "Los Angeles")[0]

        if len(houston_idx) > 0:
            target_class = houston_idx[0]
            class_name = "Houston"
        else:
            target_class = 0
            class_name = city_labels[0]

        sv_class = shap_values[target_class]  # (n_samples, n_features)
    else:
        sv_class = shap_values
        class_name = "overall"

    # Create a beeswarm-like scatter for top 6 features
    top_feats_idx = np.argsort(np.mean(np.abs(sv_class), axis=0))[-6:][::-1]

    fig_bee = make_subplots(rows=len(top_feats_idx), cols=1, vertical_spacing=0.02)
    for row, fi in enumerate(top_feats_idx):
        feat_name = feature_display.get(all_features[fi], all_features[fi])
        feat_vals = X_shap[:, fi]
        shap_v = sv_class[:, fi]

        # Normalize feature values for color
        fmin, fmax = feat_vals.min(), feat_vals.max()
        if fmax > fmin:
            feat_norm = (feat_vals - fmin) / (fmax - fmin)
        else:
            feat_norm = np.zeros_like(feat_vals)

        fig_bee.add_trace(go.Scatter(
            x=shap_v,
            y=np.random.normal(0, 0.1, len(shap_v)),
            mode="markers",
            marker=dict(
                size=4, color=feat_norm,
                colorscale="RdBu_r", opacity=0.6,
                showscale=(row == 0),
                colorbar=dict(title="Feature Value", len=0.3, y=0.85) if row == 0 else None,
            ),
            showlegend=False,
            hovertext=[f"{feat_name}={v:.1f}, SHAP={s:.3f}" for v, s in zip(feat_vals, shap_v)],
        ), row=row + 1, col=1)
        fig_bee.update_yaxes(
            title_text=feat_name[:15], row=row + 1, col=1,
            showticklabels=False,
        )

    fig_bee.update_xaxes(title_text=f"SHAP Value (impact on {class_name} prediction)",
                          row=len(top_feats_idx), col=1)
    fig_bee.update_layout(
        height=120 * len(top_feats_idx) + 100,
        template="plotly_white",
        title_text=f"SHAP Summary for {class_name} Class",
        margin=dict(l=120, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_bee, use_container_width=True)

    # Climate insights
    if isinstance(shap_values, list) and len(houston_idx) > 0 and len(la_idx) > 0:
        houston_sv = shap_values[houston_idx[0]]
        la_sv = shap_values[la_idx[0]]

        humidity_idx = all_features.index("relative_humidity_pct")

        # Average SHAP for humidity when predicting Houston vs LA
        humidity_shap_houston = np.mean(houston_sv[:, humidity_idx])
        humidity_shap_la = np.mean(la_sv[:, humidity_idx])

        insight_box(
            f"Here is where SHAP tells you something genuinely interesting about "
            f"climate. High humidity pushes predictions toward Houston "
            f"(SHAP = {humidity_shap_houston:+.3f}), while low humidity pushes "
            f"toward Los Angeles (SHAP = {humidity_shap_la:+.3f}). This makes "
            "perfect physical sense: Houston is a humid Gulf Coast city where the "
            "air feels like warm soup, while LA has a dry Mediterranean climate "
            "where the desert is always nearby. The model independently discovered "
            "what every weather forecaster already knows."
        )

else:
    # Fallback: show model-based importance comparison
    st.subheader("Feature Importance Insights (without SHAP)")

    st.markdown(
        "Without the SHAP package, we can still gain directional insights by "
        "examining how feature values differ across correctly classified cities. "
        "This is a poor man's SHAP, but it reveals the same basic patterns."
    )

    preds = model.predict(X_test)
    correct_mask = preds == y_test

    insights_data = []
    for ci, city_name in enumerate(city_labels):
        city_mask = (y_test == ci) & correct_mask
        if city_mask.sum() > 10:
            city_means = X_test[city_mask].mean(axis=0)
            all_means = X_test[correct_mask].mean(axis=0)
            for fi, feat in enumerate(all_features):
                diff = city_means[fi] - all_means[fi]
                insights_data.append({
                    "City": city_name,
                    "Feature": feature_display.get(feat, feat),
                    "Mean Diff from Average": round(diff, 3),
                })

    if insights_data:
        ins_df = pd.DataFrame(insights_data)
        # Show as heatmap
        pivot = ins_df.pivot(index="Feature", columns="City", values="Mean Diff from Average")
        fig_hm = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale="RdBu_r", zmid=0,
        ))
        apply_common_layout(fig_hm, "Feature Deviations by City (vs Global Mean)", 500)
        st.plotly_chart(fig_hm, use_container_width=True)

        insight_box(
            "Houston stands out with above-average humidity and dew point -- "
            "it is the city the model identifies by its sticky, humid weather. "
            "Los Angeles shows below-average humidity, consistent with its dry "
            "Mediterranean climate. These distinctive feature profiles are the "
            "'fingerprints' the model uses to classify cities."
        )

# ── Section 4: All Three Methods Compared ────────────────────────────────────
st.header("4. Comparing All Methods")

st.markdown(
    "The ultimate sanity check. We normalize all importance scores to [0, 1] "
    "and plot them side-by-side. Features where all methods agree are ones you "
    "can be confident about. Features where methods disagree are telling you "
    "something interesting -- usually about correlations or nonlinearities that "
    "some methods capture and others miss."
)

def normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn) if mx > mn else arr

comp_methods = pd.DataFrame({
    "Feature": [feature_display.get(f, f) for f in all_features],
    "MDI": normalize(mdi),
    "Permutation": normalize(perm_result.importances_mean),
})
if shap_available:
    comp_methods["SHAP"] = normalize(shap_importance)

comp_melted = comp_methods.melt(id_vars="Feature", var_name="Method", value_name="Normalized Importance")

fig_comp = px.bar(
    comp_melted, x="Feature", y="Normalized Importance", color="Method",
    barmode="group",
    color_discrete_sequence=["#2E86C1", "#E63946", "#2A9D8F"],
    title="Feature Importance: Method Comparison",
)
fig_comp.update_layout(xaxis_tickangle=-45)
apply_common_layout(fig_comp, height=500)
st.plotly_chart(fig_comp, use_container_width=True)

warning_box(
    "Feature importance does not imply causation. A feature may rank as important "
    "because it *correlates* with the true causal variable, not because it directly "
    "causes the outcome. Humidity may be important for classifying Houston not because "
    "humidity *causes* Houston to be Houston, but because Houston's proximity to the "
    "Gulf of Mexico causes both the city's identity and its humidity. If you need "
    "causal claims, you need causal inference methods, not feature importance."
)

code_example("""
import shap
from sklearn.inspection import permutation_importance

# MDI (built-in for tree models)
mdi_importance = model.feature_importances_

# Permutation importance (model-agnostic)
perm = permutation_importance(model, X_test, y_test, n_repeats=10)
perm_importance = perm.importances_mean

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
""")

# ── Quiz ─────────────────────────────────────────────────────────────────────
st.divider()
quiz(
    "Which feature importance method works with ANY model type, not just trees?",
    [
        "Mean Decrease in Impurity (MDI)",
        "Gini Importance",
        "Permutation Importance",
        "Split Count",
    ],
    2,
    "Permutation importance only requires the ability to make predictions and "
    "evaluate accuracy. It works with literally any model -- trees, neural "
    "networks, SVMs, linear models, even that custom model your colleague built "
    "in Excel. It is the universal adapter of feature importance.",
    key="fi_quiz",
)

# ── Takeaways ────────────────────────────────────────────────────────────────
st.divider()
takeaways([
    "**MDI** is fast and free with tree models, but biased toward high-cardinality and correlated features. Use it as a quick first look, not as gospel.",
    "**Permutation importance** is model-agnostic and unbiased -- it directly measures 'what happens when I break this feature?' But it is slower because it requires multiple evaluation passes.",
    "**SHAP** is the gold standard. It provides both global importance and per-prediction explanations, reveals the *direction* of effect, and has strong theoretical foundations in game theory.",
    "SHAP reveals climate fingerprints: high humidity pushes predictions toward Houston, low humidity toward LA. This is the kind of directional insight the other methods cannot provide.",
    "Always compare multiple methods. Features that consistently rank high across all three are truly important. Disagreements are diagnostic opportunities.",
    "Feature importance is not causation. Correlation-based importance can mislead if you interpret it as 'this feature causes the outcome.' Be honest about what the numbers do and do not tell you.",
])
