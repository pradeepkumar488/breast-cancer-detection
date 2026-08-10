"""
Breast Cancer Detection System - Web App (Streamlit)
-------------------------------------------------------
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)

st.set_page_config(page_title="Breast Cancer Detection System", layout="wide")

RANDOM_STATE = 42

# ---------------------------------------------------------------
# LOAD + TRAIN (cached so it only runs once, not on every click)
# ---------------------------------------------------------------
@st.cache_resource
def load_and_train():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selector = SelectKBest(score_func=f_classif, k=15)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)
    selected_features = X.columns[selector.get_support()]

    models = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM (RBF Kernel)": SVC(probability=True, random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)
        y_proba = model.predict_proba(X_test_sel)[:, 1]
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_proba),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    best_name = max(results, key=lambda k: results[k]["f1"])
    return data, X, y, X_train, X_test, y_train, y_test, scaler, selector, selected_features, models, results, best_name


data, X, y, X_train, X_test, y_train, y_test, scaler, selector, selected_features, models, results, best_name = load_and_train()
best = results[best_name]

# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
st.title("🩺 Breast Cancer Detection System")
st.caption("Python · Scikit-learn · Machine Learning · Data Preprocessing")
st.write(
    "This app classifies tumors as **Benign** or **Malignant** using the "
    "Breast Cancer Wisconsin (Diagnostic) dataset (569 patients, 30 measurements each)."
)

tab1, tab2, tab3 = st.tabs(["📊 Model Results", "🔮 Try a Live Prediction", "📈 Charts"])

# ---------------------------------------------------------------
# TAB 1: MODEL RESULTS
# ---------------------------------------------------------------
with tab1:
    st.subheader("Model Comparison")
    comp_df = pd.DataFrame({
        name: {
            "Accuracy": f"{r['accuracy']:.2%}",
            "Precision": f"{r['precision']:.2%}",
            "Recall": f"{r['recall']:.2%}",
            "F1 Score": f"{r['f1']:.2%}",
            "ROC-AUC": f"{r['auc']:.4f}",
        } for name, r in results.items()
    }).T
    st.dataframe(comp_df, use_container_width=True)

    st.success(f"**Best model: {best_name}** (F1 Score = {best['f1']:.2%})")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{best['accuracy']:.2%}")
        st.metric("Precision", f"{best['precision']:.2%}")
    with col2:
        st.metric("Recall", f"{best['recall']:.2%}")
        st.metric("ROC-AUC", f"{best['auc']:.4f}")

    st.subheader("Top 15 Selected Features")
    st.write(", ".join(selected_features))

# ---------------------------------------------------------------
# TAB 2: LIVE PREDICTION (the "AI" interactive part)
# ---------------------------------------------------------------
with tab2:
    st.subheader("Enter tumor measurements to get a live prediction")
    st.write("Adjust the sliders (defaults are a real patient sample from the test set) and click **Predict**.")

    sample_idx = st.slider("Load a test-set patient sample as a starting point", 0, len(X_test) - 1, 0)
    sample_row = X_test.iloc[sample_idx]

    input_vals = {}
    cols = st.columns(3)
    for i, feature in enumerate(selected_features):
        col = cols[i % 3]
        min_v = float(X[feature].min())
        max_v = float(X[feature].max())
        default_v = float(sample_row[feature])
        input_vals[feature] = col.slider(feature, min_v, max_v, default_v)

    if st.button("🔍 Predict", type="primary"):
        # Build a full 30-feature row (unselected features use the sample's original values)
        full_row = sample_row.copy()
        for feature, val in input_vals.items():
            full_row[feature] = val

        row_df = pd.DataFrame([full_row])
        row_scaled = scaler.transform(row_df)
        row_selected = selector.transform(row_scaled)

        pred = best["model"].predict(row_selected)[0]
        proba = best["model"].predict_proba(row_selected)[0]

        label = "Benign" if pred == 1 else "Malignant"
        if pred == 1:
            st.success(f"### Prediction: {label} ✅")
        else:
            st.error(f"### Prediction: {label} ⚠️")

        c1, c2 = st.columns(2)
        c1.metric("Malignant confidence", f"{proba[0]:.1%}")
        c2.metric("Benign confidence", f"{proba[1]:.1%}")

        actual = "Benign" if y_test.iloc[sample_idx] == 1 else "Malignant"
        st.caption(f"Actual label for this original sample: {actual}")

# ---------------------------------------------------------------
# TAB 3: CHARTS
# ---------------------------------------------------------------
with tab3:
    st.subheader("Confusion Matrix & ROC Curve")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm = confusion_matrix(y_test, best["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"], ax=axes[0])
    axes[0].set_title(f"Confusion Matrix - {best_name}")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_title("ROC Curves")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(fontsize=8)

    st.pyplot(fig)

    st.subheader("Feature Importance (Random Forest)")
    rf = models["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=selected_features).sort_values()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax2, color="teal")
    st.pyplot(fig2)