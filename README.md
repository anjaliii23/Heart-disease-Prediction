# ❤️ Heart Disease Prediction using Machine Learning

![Project Banner](images/heart_banner.jpg)

## 📊 Project Overview

This project predicts the likelihood of heart disease in patients using various clinical features such as age, cholesterol, chest pain type, and resting blood pressure.
It employs a **Random Forest Classifier** integrated into a **Pipeline** with preprocessing via `StandardScaler()` and parameter optimization using `RandomizedSearchCV`.

> 🎯 **Goal:** Build an accurate and interpretable model for early heart disease risk detection.

---

## 📁 Dataset Description

| Attribute | Description                                                       |
| --------- | ----------------------------------------------------------------- |
| age       | Age of the patient                                                |
| sex       | Gender (1 = Male, 0 = Female)                                     |
| cp        | Chest pain type                                                   |
| trestbps  | Resting blood pressure (mm Hg)                                    |
| chol      | Serum cholesterol (mg/dl)                                         |
| fbs       | Fasting blood sugar > 120 mg/dl                                   |
| restecg   | Resting electrocardiographic results                              |
| thalach   | Maximum heart rate achieved                                       |
| exang     | Exercise-induced angina                                           |
| oldpeak   | ST depression induced by exercise                                 |
| slope     | Slope of the peak exercise ST segment                             |
| ca        | Number of major vessels colored by fluoroscopy                    |
| thal      | Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect) |
| target    | Heart disease (1 = yes, 0 = no)                                   |

**Rows:** 303
**Columns:** 14
**Missing Values:** None

---

## ⚙️ Model Workflow

### 1️⃣ Data Preprocessing

* Standardized features using **StandardScaler**
* Split into 70% training and 30% testing sets
* Verified class balance

### 2️⃣ Model Building

Pipeline structure:

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
```

### 3️⃣ Hyperparameter Tuning

RandomizedSearchCV parameters:

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

**Best Parameters:**

```python
{'n_estimators': 200,
 'min_samples_split': 2,
 'min_samples_leaf': 4,
 'max_features': 'sqrt',
 'max_depth': 20,
 'bootstrap': True}
```

---

## 📈 Model Evaluation

| Metric    | Score    |
| --------- | -------- |
| Accuracy  | **0.82** |
| Precision | **0.83** |
| Recall    | **0.86** |
| F1 Score  | **0.84** |

**Confusion Matrix:**

```
[[32,  9],
 [ 7, 43]]
```

![Confusion Matrix](images/confusion_matrix.png)

---

## 🔍 Data Insights

* **Male Patients:** 68.32%
* **Female Patients:** 31.68%
* **Heart Disease Cases:** 54.46%
* **No Heart Disease:** 45.54%

![Gender Distribution](images/gender_distribution.png)
![Target Distribution](images/target_distribution.png)

### 🔗 Feature Correlation with Target

| Feature | Correlation |
| ------- | ----------- |
| exang   | 0.44        |
| cp      | 0.43        |
| oldpeak | 0.43        |
| thalach | 0.42        |
| ca      | 0.39        |

![Correlation Heatmap](images/correlation_heatmap.png)

### 🧩 Feature Importance (Top 5)

![Feature Importance](images/feature_importance.png)

---

## 🧠 Technologies Used

* Python 3.11
* pandas
* numpy
* scikit-learn (v1.4.1.post1)
* matplotlib / seaborn

---

## 💡 Key Learnings

* Using `Pipeline` helps streamline scaling and training.
* Random Forest achieved consistent performance with minimal overfitting.
* The dataset showed a slightly higher occurrence of heart disease among males.

---

## 🚀 Future Improvements

* Integrate the model into a **Flask web app** for real-time predictions.
* Add **SHAP or LIME** for model interpretability.
* Create a **dashboard in Power BI or Streamlit** for interactive visualization.

---

## 🖼️ Suggested Folder Structure

```
heart-disease-prediction/
│
├── data/
│   └── heart.csv
│
├── images/
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── gender_distribution.png
│   ├── target_distribution.png
│   ├── feature_importance.png
│   └── heart_banner.jpg
│
├── notebook/
│   └── heart_disease_model.ipynb
│
├── README.md
└── requirements.txt
```
