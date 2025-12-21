# UCI Heart Disease Modeling

This repository contains the final project for **Data Science Fundamentals with Python**, focused on analyzing and modeling heart disease data using classical machine learning techniques.

## 📌 Project Overview
The goal of this project is to explore, preprocess, and model the **UCI Cleveland Heart Disease dataset** to:
- Predict the presence of heart disease (classification)
- Predict serum cholesterol levels (regression)
- Reduce dimensionality using PCA
- Group patients based on health profiles using clustering

The project demonstrates an end-to-end data science workflow including EDA, preprocessing, modeling, evaluation, and interpretation.

---

## 📊 Dataset
- **Source:** UCI Cleveland Heart Disease Dataset  
- **Samples:** 303 patients  
- **Features:** 13 clinical attributes  
- **Target:** `num`
  - `0` → No heart disease  
  - `1–4` → Presence of heart disease (converted to binary)

---

## 🧪 Tasks Performed

### 1. Exploratory Data Analysis & Preprocessing
- Data inspection using `head()`, `info()`, and `describe()`
- Missing value imputation using column-wise median
- Binary conversion of target variable
- Feature normalization using `StandardScaler`

### 2. Heart Disease Prediction (Classification)
Models implemented:
- Logistic Regression
- Random Forest Classifier

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### 3. Cholesterol Level Prediction (Regression)
- Multiple Linear Regression model
- Correlation analysis and heatmap visualization
- Identification of features most correlated with serum cholesterol

### 4. Principal Component Analysis (PCA)
- Dimensionality reduction with 95% variance retained
- Explained variance ratio visualization
- Reduced feature space for downstream tasks

### 5. Patient Grouping using K-Means Clustering
- K-Means applied on PCA-reduced data
- Optimal cluster selection using:
  - Elbow Method
  - Silhouette Score
- Cluster visualization using first two PCA components

---

## 🛠️ Tools & Libraries
- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn

---

## ✅ Outcome
This project provides practical experience with supervised and unsupervised learning techniques applied to real-world clinical data, highlighting how data science can aid in healthcare-related decision making.

---
