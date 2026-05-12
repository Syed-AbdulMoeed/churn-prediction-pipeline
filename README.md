# Customer Churn Prediction Pipeline

A machine learning project to predict customer churn from structured CRM/telecom data. Built as a full end-to-end pipeline — from raw data to a deployed interactive web application.

**Live demo:** *(link added on completion)*

---

## Results

| Model | ROC-AUC | F1 Score | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.84 | 0.74 | 0.5 | 0.79 |
| Random Forest | — | — | — | — |
| XGBoost (final) | — | — | — | — |

*Metrics to be filled in during Week 3.*
*Precision, Recall, F1 Score is for Churn(1) Predicition*

---

## Project overview

Customer churn — when a customer stops using a service — is one of the most common and costly problems for subscription-based businesses. Predicting which customers are likely to churn allows companies to intervene early with retention offers or support.

This project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle. The dataset contains 7,043 customer records with 20 features including contract type, tenure, monthly charges, and usage patterns. The target variable is binary: churned or not.

**Business question:** Given a customer's profile and usage behaviour, what is the probability they will churn in the next billing period?

---

## Repository structure

```
churn-prediction/
│
├── data/
│   ├── raw/                  # Original downloaded dataset (not modified)
│   └── processed/            # Cleaned and feature-engineered data
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_modelling.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── features.py           # Feature engineering functions
│   ├── train.py              # Model training script
│   └── predict.py            # Inference logic
│
├── models/
│   └── model.pkl             # Saved best model + pipeline
│
├── reports/
│   └── project_report.pdf    # Final write-up with methodology and results
│
├── app.py                    # Streamlit web application
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Exploratory data analysis
- Distribution of churn vs non-churn (~27% churn rate — imbalanced)
- Missing value analysis (`TotalCharges` has nulls for new customers)
- Feature correlation with churn label
- Key finding: Contract type, tenure, and monthly charges are strong churn predictors

### 2. Feature engineering & preprocessing
- Encoded categorical variables using `OneHotEncoder`
- Scaled numerical features with `StandardScaler`
- Addressed class imbalance using `class_weight='balanced'` and evaluated SMOTE
- Built a reproducible `sklearn Pipeline` to prevent data leakage

### 3. Modelling
Trained three models in order of complexity:
1. **Logistic Regression** — interpretable baseline
2. **Random Forest** — captures non-linear relationships
3. **XGBoost** — gradient boosted trees, best performance on tabular data

Hyperparameters tuned with `GridSearchCV` on `n_estimators`, `max_depth`, and `learning_rate`.

### 4. Evaluation
- Primary metric: **ROC-AUC** (handles class imbalance better than accuracy)
- Secondary: F1 score, precision, recall
- Visualised: confusion matrix, ROC curve, SHAP feature importance

### 5. Deployment
Interactive Streamlit app that accepts a CSV of customer records and returns churn probability scores per customer, along with the top contributing features per prediction.

---

## Key findings

*(To be filled in after Week 3)*

- Most important churn predictors: ...
- Contract type had the highest impact because ...
- Model performs best on customers with tenure > X months

---

## How to run locally

**1. Clone the repository**
```bash
git clone https://github.com/Syed-AbdulMoeed/churn-prediction-pipeline.git
cd churn-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Tech stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| pandas | Data manipulation |
| scikit-learn | Preprocessing, modelling, evaluation |
| XGBoost | Gradient boosted model |
| SHAP | Model explainability |
| Streamlit | Web application |
| matplotlib / seaborn | Visualisation |

---

## What I'd do next (production considerations)

- **Retrain pipeline**: Set up scheduled retraining as new customer data arrives
- **Threshold tuning**: Adjust classification threshold based on business cost of false negatives vs false positives
- **Monitoring**: Track model performance drift over time using Evidently or similar
- **API endpoint**: Wrap the model in a FastAPI endpoint for integration with CRM systems

---

## Author

**Syed Muhammad Abdul Moeed**  
BSc Artificial Intelligence — Deggendorf Institute of Technology (2nd semester)  
[LinkedIn](https://linkedin.com/in/muhammad-abdul-moeed-syed-143b5330b) · [GitHub](https://github.com/Syed-AbdulMoeed)