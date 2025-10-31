# ❤️ Heart Disease Prediction using Machine Learning

## 📊 Project Overview

This project predicts the likelihood of heart disease in patients based on various medical attributes such as age, cholesterol levels, chest pain type, and resting blood pressure. It uses a **Random Forest Classifier** integrated into a **machine learning pipeline** with preprocessing steps for scaling and parameter tuning.

The goal is to build an accurate, interpretable, and efficient model to assist in early identification of heart disease risk.

---

## 📁 Dataset

The dataset used in this project contains **303 patient records** with **14 attributes**, including:

* `age` – Age of the patient
* `sex` – Gender (1 = male, 0 = female)
* `cp` – Chest pain type
* `trestbps` – Resting blood pressure
* `chol` – Serum cholesterol (mg/dl)
* `fbs` – Fasting blood sugar > 120 mg/dl
* `restecg` – Resting electrocardiographic results
* `thalach` – Maximum heart rate achieved
* `exang` – Exercise-induced angina
* `oldpeak` – ST depression induced by exercise
* `slope` – Slope of the peak exercise ST segment
* `ca` – Number of major vessels colored by fluoroscopy
* `thal` – Thalassemia (defect type)
* `target` – Diagnosis of heart disease (1 = disease, 0 = no disease)

There are **no missing values** in the dataset.

---

## ⚙️ Model Workflow

### 1. **Data Preprocessing**

* Scaled numerical features using `StandardScaler`
* Split dataset into training and testing sets (70:30 ratio)
* Ensured balanced class distribution

### 2. **Model Building**

Used a **Pipeline** combining:

* `StandardScaler()` – for feature normalization
* `RandomForestClassifier()` – for classification

### 3. **Hyperparameter Tuning**

Optimized using **RandomizedSearchCV** with parameters:

```python
{
 'n_estimators': [100, 200, 500],
 'max_features': ['auto', 'sqrt'],
 'max_depth': [5, 10, 20, None],
 'min_samples_split': [2, 5, 10],
 'min_samples_leaf': [1, 2, 4],
 'bootstrap': [True, False]
}
```

Best parameters obtained:

```python
{'n_estimators': 200,
 'min_samples_split': 2,
 'min_samples_leaf': 4,
 'max_features': 'sqrt',
 'max_depth': 20,
 'bootstrap': True}
```

---

## 📈 Model Performance

| Metric        | Score |
| ------------- | ----- |
| **Accuracy**  | 0.82  |
| **Precision** | 0.83  |
| **Recall**    | 0.86  |
| **F1-Score**  | 0.84  |

Confusion Matrix:

```
[[32,  9],
 [ 7, 43]]
```

---

## 🔍 Key Insights

* **54.46%** of patients had heart disease.
* Higher correlation observed between target and:

  * Exercise-induced angina (`exang`)
  * Chest pain type (`cp`)
  * Oldpeak
  * Maximum heart rate (`thalach`)
* Male patients accounted for **68.32%** of the dataset.

---

## 🧠 Technologies Used

* Python 3.11
* pandas
* numpy
* scikit-learn (v1.4.1.post1)
* matplotlib / seaborn (for visualization)

---

## 📬 Results Summary

The model achieved an **82% accuracy** in predicting heart disease and shows strong potential as a diagnostic support tool. Further improvement can be achieved by:

* Using ensemble techniques (XGBoost, Gradient Boosting)
* Performing feature engineering and correlation-based feature selection
* Expanding dataset size for better generalization

---

## 💡 Future Work

* Integrate the model into a **Flask web app** for real-time prediction.
* Add **SHAP or LIME** explanations for model interpretability.
* Build an interactive dashboard for visualization.

---

