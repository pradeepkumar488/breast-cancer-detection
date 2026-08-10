"""
Breast Cancer Detection System
--------------------------------
Tech: Python, Scikit-learn, Machine Learning, Data Preprocessing

Classifies tumors as Benign or Malignant using the Breast Cancer
Wisconsin (Diagnostic) dataset. Covers: data preprocessing,
feature selection, model training, and evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

print("Dataset shape:", X.shape)
print("Class distribution:\n", y.value_counts().rename({0: "Malignant", 1: "Benign"}))

# ---------------------------------------------------------------
# 2. DATA PREPROCESSING
# ---------------------------------------------------------------
print("\nMissing values per column:", X.isnull().sum().sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 3. FEATURE SELECTION
# ---------------------------------------------------------------
selector = SelectKBest(score_func=f_classif, k=15)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

selected_features = X.columns[selector.get_support()]
print("\nTop 15 selected features:")
print(list(selected_features))

# ---------------------------------------------------------------
# 4. MODEL TRAINING
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "SVM (RBF Kernel)": SVC(probability=True, random_state=RANDOM_STATE),
}

results = {}

for name, model in models.items():
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    y_proba = model.predict_proba(X_test_selected)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5)

    results[name] = {
        "model": model, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "auc": auc,
        "cv_mean": cv_scores.mean(), "y_pred": y_pred, "y_proba": y_proba
    }

    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

best_model_name = max(results, key=lambda k: results[k]["f1"])
best = results[best_model_name]
print(f"\n>>> Best model: {best_model_name} (F1 = {best['f1']:.4f}) <<<")
print("\nClassification Report (Best Model):")
print(classification_report(y_test, best["y_pred"], target_names=["Malignant", "Benign"]))

# ---------------------------------------------------------------
# 5. VISUALIZATIONS
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Malignant", "Benign"],
            yticklabels=["Malignant", "Benign"], ax=axes[0, 0])
axes[0, 0].set_title(f"Confusion Matrix - {best_model_name}")
axes[0, 0].set_ylabel("Actual")
axes[0, 0].set_xlabel("Predicted")

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[0, 1].set_title("ROC Curves")
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].legend(loc="lower right", fontsize=8)

comp_df = pd.DataFrame({k: {"Accuracy": v["accuracy"], "F1": v["f1"], "AUC": v["auc"]}
                         for k, v in results.items()}).T
comp_df.plot(kind="bar", ax=axes[1, 0], rot=20)
axes[1, 0].set_title("Model Comparison")
axes[1, 0].set_ylim(0.85, 1.0)
axes[1, 0].legend(loc="lower right", fontsize=8)

if "Random Forest" in models:
    rf = models["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=selected_features)
    importances.sort_values().plot(kind="barh", ax=axes[1, 1], color="teal")
    axes[1, 1].set_title("Feature Importance (Random Forest)")

plt.tight_layout()
plt.savefig("breast_cancer_results.png", dpi=150)
print("\nSaved visualization to breast_cancer_results.png")

# ---------------------------------------------------------------
# 6. SAMPLE PREDICTION DEMO
# ---------------------------------------------------------------
sample = X_test.iloc[[0]]
sample_scaled = scaler.transform(sample)
sample_selected = selector.transform(sample_scaled)
pred = best["model"].predict(sample_selected)[0]
proba = best["model"].predict_proba(sample_selected)[0]

print("\n--- Sample Prediction ---")
print(f"Predicted class: {'Benign' if pred == 1 else 'Malignant'}")
print(f"Confidence: Malignant={proba[0]:.2%}, Benign={proba[1]:.2%}")
print(f"Actual class: {'Benign' if y_test.iloc[0] == 1 else 'Malignant'}")