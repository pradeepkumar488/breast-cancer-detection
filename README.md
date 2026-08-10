# 🩺 Breast Cancer Detection System

A machine learning system that classifies tumors as **Benign** or **Malignant** using the Breast Cancer Wisconsin (Diagnostic) dataset.

## 🔧 Tech Stack
- Python
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn
- Streamlit (web app)

## 📌 Project Overview
This project applies data preprocessing, feature selection, and multiple classification algorithms to predict whether a tumor is benign or malignant, supporting early diagnosis.

## ⚙️ Features
- Data preprocessing (scaling, train/test split)
- Feature selection using SelectKBest (ANOVA F-value)
- Multiple models trained and compared:
  - Logistic Regression
  - Random Forest
  - SVM (RBF Kernel)
- Evaluation using Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Confusion matrix and ROC curve visualizations
- Interactive web app (Streamlit) with live prediction

## 📊 Results
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 96.5% | 97.2% | 97.2% | 97.2% | 0.991 |
| Random Forest | 95.6% | 95.9% | 97.2% | 96.6% | 0.991 |
| SVM (RBF Kernel) | 94.7% | 95.8% | 95.8% | 95.8% | 0.991 |

**Best Model: Logistic Regression**

## 🚀 How to Run

### 1. Install dependencies
pip install scikit-learn pandas matplotlib seaborn streamlit
### 2. Run the script (terminal output + chart)
python breast_cancer_detection.py
### 3. Run the web app (interactive browser demo)
python -m streamlit run app.py
## 📁 Dataset
Breast Cancer Wisconsin (Diagnostic) Dataset — built into scikit-learn (`sklearn.datasets.load_breast_cancer`), 569 samples, 30 features.

## 👤 Author
Pradeepa Kumara
