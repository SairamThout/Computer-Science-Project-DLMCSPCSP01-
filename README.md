# Predicting Diabetes Using Machine Learning: A Comparative Analysis of Classification Algorithms

##  Project Overview
This project investigates the use of **machine learning algorithms** to predict diabetes using the **Pima Indians Diabetes Dataset**. The study compares baseline and advanced supervised learning classifiers, focusing on model effectiveness for **early diagnosis** and **decision support in healthcare**.

---

## Objectives
- To apply **Logistic Regression, Random Forest, and Gradient Boosting** to predict diabetes.  
- To perform **data preprocessing, exploratory data analysis (EDA), and feature scaling**.  
- To evaluate models using **accuracy, precision, recall, F1-score, and ROC-AUC**.  
- To ensure **robustness** through cross-validation, unit testing, and workflow validation.  
- To provide insights into the applicability of machine learning in **real-world healthcare decision-making**.



##  Dataset
- **Source**: [Pima Indians Diabetes Dataset (UCI Machine Learning Repository)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- **Records**: 768 patient observations  
- **Features**:  
  - Glucose levels  
  - BMI  
  - Blood Pressure  
  - Age  
  - Number of Pregnancies  
  - Insulin  
  - Skin Thickness  
  - Diabetes Pedigree Function  
- **Target**: `Outcome` (1 = diabetic, 0 = non-diabetic)  


##  Tools & Libraries
- **Python** (Jupyter Notebook)  
- **pandas, numpy** → Data handling  
- **matplotlib, seaborn** → Visualization (EDA, ROC curves)  
- **scikit-learn** → Model training, evaluation, preprocessing  
- **pytest** → Unit testing  
- **joblib** → Model saving and reproducibility  

---

##  Methodology
1. **Data Loading & Overview** – Load dataset and check for missing values.  
2. **Exploratory Data Analysis (EDA)** – Histograms, boxplots, correlations.  
3. **Preprocessing** – Median imputation for missing values, Z-score scaling, stratified train-test split.  
4. **Model Training** – Logistic Regression (baseline), Random Forest, Gradient Boosting.  
5. **Evaluation** – Accuracy, Precision, Recall, F1-score, ROC-AUC, Cross-validation.  
6. **Testing** – Unit tests for preprocessing pipeline and workflow validation.  

---

##  Implementation Workflow

Data Loading → Preprocessing → Train-Test Split → Model Training → Evaluation → Cross-Validation → Unit Testing → Workflow Validation
