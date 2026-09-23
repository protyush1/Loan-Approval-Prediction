# 🏦 Loan Approval Prediction

A complete **Data Analytics and Machine Learning** project that predicts loan approval outcomes using real applicant and financial data. Built with Python and Streamlit.

---

## 📋 Project Structure

| Section | Description |
|---------|-------------|
| **Problem Statement** | Defines the business problem and objective |
| **Dataset Overview** | Data inspection, statistics, and quality checks |
| **Steps Used** | End-to-end methodology walkthrough |
| **Live Analysis** | Interactive EDA with charts and findings |
| **ML Models** | Training, evaluation, and comparison of 4 models |
| **Prediction** | Live loan approval prediction form |
| **Business Decisions** | Data-driven actionable recommendations |
| **Conclusion** | Summary of findings and model results |

---

## 📦 Dataset

**File:** `loan_approval_dataset.csv`(https://www.kaggle.com/code/jayrdixit/loan-approval-prediction-dataset/input)  
**Records:** 4,269 loan applications  
**Features:** 13 columns (12 features + 1 target)

| Column | Type | Description |
|--------|------|-------------|
| `loan_id` | int | Unique identifier (excluded from modeling) |
| `no_of_dependents` | int | Number of dependents (0–5) |
| `education` | str | Graduate / Not Graduate |
| `self_employed` | str | Yes / No |
| `income_annum` | int | Annual income (₹) |
| `loan_amount` | int | Requested loan amount (₹) |
| `loan_term` | int | Loan term in years (2–20) |
| `cibil_score` | int | Credit score (300–900) |
| `residential_assets_value` | int | Residential asset value (₹) |
| `commercial_assets_value` | int | Commercial asset value (₹) |
| `luxury_assets_value` | int | Luxury asset value (₹) |
| `bank_asset_value` | int | Bank / savings value (₹) |
| `loan_status` | str | **Target** — Approved / Rejected |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher

### Installation

1. **Clone / download** this project folder.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`

---

## 🛠️ Technology Stack

| Library | Purpose |
|---------|---------|
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical operations |
| **Matplotlib / Seaborn** | Visualization |
| **Scikit-learn** | Preprocessing, model training, evaluation |
| **XGBoost** | Gradient boosting model |
| **Streamlit** | Interactive web application |
| **ReportLab** | PDF report generation |

---

## 🤖 Models Trained

1. **Logistic Regression** — Interpretable baseline (with StandardScaler)
2. **Decision Tree** — Rule-based, max depth 8
3. **Random Forest** — Ensemble, 200 estimators
4. **XGBoost** — Gradient boosting, 200 estimators

Model selection is based on **F1-Score** on the 20% stratified test set.

---

## 📊 Key Findings

- CIBIL score is the most important predictor of loan approval.
- Approved applicants have significantly higher average CIBIL scores.
- Higher annual income and total asset values positively correlate with approval.
- Education (Graduate) and salaried employment show slightly higher approval rates.
- Number of dependents has a mild negative influence at higher counts.

---

## 📁 Project Files

```
loan_approval_dataset.csv   — Input dataset
app.py                      — Main Streamlit application
requirements.txt            — Python dependencies
README.md                   — This file
report.pdf                  — Project report
```

---

## 📄 License

This project is for educational and demonstration purposes.
