# ❤️ Heart Disease Prediction - Machine Learning Pipeline

A complete end-to-end machine learning pipeline to predict the presence of heart disease using the UCI Heart Disease dataset.

---

## 📌 Project Overview

This project builds and evaluates several machine learning models to classify the presence of heart disease in patients. It includes the full pipeline from data preprocessing to model deployment.

---

## 🧠 Dataset

- **Source**: UCI Heart Disease Dataset  
- **Target Variable**: `target` (0 = No Heart Disease, 1 = Heart Disease)  
- **Original Features**: 13 features + target  
- **Processed File**: `./data/heart_disease_selected_features.csv`

---

## 🧰 Tools & Libraries

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Joblib  
- Jupyter Notebooks  
- Visual Studio Code  

---

## 🧪 Pipeline Steps

### 1. Data Preprocessing
- Handled missing values  
- Encoded categorical variables  
- Scaled features using `StandardScaler`  

### 2. Dimensionality Reduction (PCA)
- Applied Principal Component Analysis  
- Chose optimal components using explained variance  
- Visualized PCA scatter plot and cumulative variance  

### 3. Feature Selection
- Used multiple techniques:
  - Random Forest Feature Importance  
  - Recursive Feature Elimination (RFE)  
  - Chi-Square Test  
- Saved final selected features for modeling  

### 4. Supervised Learning
- Trained models:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Support Vector Machine (SVM)  
- Evaluated using:
  - Accuracy, Precision, Recall, F1-score  
  - ROC Curve & AUC Score  

### 5. Unsupervised Learning
- Clustering with:
  - K-Means (Elbow Method)  
  - Hierarchical Clustering (Dendrogram)  
- Compared clusters with actual labels using **Adjusted Rand Index**  

### 6. Hyperparameter Tuning
- Tuned Random Forest using `RandomizedSearchCV`  
- Evaluated best model with classification report  

### 7. Model Export
- Exported full pipeline (Scaler + Model) using `joblib`  
- ✅ Saved at: `./models/heart_disease_pipeline.pkl`

---

## 🗂️ Project Structure


Heart_Disease_Project/
├── data/
│ ├── heart_disease.csv
│ └── heart_disease_selected_features.csv
├── models/
│ └── final_model.pkl
├── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_pca_analysis.ipynb
│ ├── 03_feature_selection.ipynb
│ ├── 04_supervised_learning.ipynb
│ ├── 05_unsupervised_learning.ipynb
│ ├── 06_hyperparameter_tuning.ipynb
│ └── 07_model_export.ipynb
├── results/
│ └── evaluation_metrics.txt
├── requirements.txt
├── README.md
└── .gitignore


---

## 📦 Setup Instructions

Install all required dependencies using:

```bash
pip install -r requirements.txt
