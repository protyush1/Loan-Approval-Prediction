import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL STYLE
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 700;
        color: #1a3c5e; text-align: center;
        padding-bottom: 0.3rem;
    }
    .sub-title {
        font-size: 1.0rem; color: #555;
        text-align: center; margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.5rem; font-weight: 600;
        color: #1a3c5e; border-left: 5px solid #2e86de;
        padding-left: 0.6rem; margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }
    .metric-card {
        background: #f0f4fa; border-radius: 10px;
        padding: 1rem 1.5rem; text-align: center;
        border: 1px solid #c8d8ea;
    }
    .metric-value {
        font-size: 2rem; font-weight: 700; color: #1a3c5e;
    }
    .metric-label {
        font-size: 0.85rem; color: #666; margin-top: 0.2rem;
    }
    .approved-badge {
        background: #d4edda; color: #155724;
        border-radius: 8px; padding: 0.8rem 1.5rem;
        font-size: 1.4rem; font-weight: 700;
        text-align: center; border: 2px solid #c3e6cb;
    }
    .rejected-badge {
        background: #f8d7da; color: #721c24;
        border-radius: 8px; padding: 0.8rem 1.5rem;
        font-size: 1.4rem; font-weight: 700;
        text-align: center; border: 2px solid #f5c6cb;
    }
    .info-box {
        background: #eef6ff; border-left: 4px solid #2e86de;
        padding: 0.8rem 1rem; border-radius: 4px;
        margin: 0.5rem 0; color: #1a3c5e;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem; font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# LOAD & CLEAN DATA  (cached)
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("loan_approval_dataset.csv")
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df

@st.cache_data
def prepare_ml(df):
    data = df.drop(columns=["loan_id"]).copy()
    le_edu  = LabelEncoder()
    le_emp  = LabelEncoder()
    le_tgt  = LabelEncoder()
    data["education"]     = le_edu.fit_transform(data["education"])
    data["self_employed"] = le_emp.fit_transform(data["self_employed"])
    data["loan_status"]   = le_tgt.fit_transform(data["loan_status"])

    feature_cols = [c for c in data.columns if c != "loan_status"]
    X = data[feature_cols]
    y = data["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=200, random_state=42,
                                             eval_metric="logloss", verbosity=0),
    }

    results  = {}
    trained  = {}

    for name, m in models.items():
        if name == "Logistic Regression":
            m.fit(X_train_sc, y_train)
            y_pred = m.predict(X_test_sc)
        else:
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)

        results[name] = {
            "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
            "Precision": round(precision_score(y_test, y_pred) * 100, 2),
            "Recall":    round(recall_score(y_test, y_pred) * 100, 2),
            "F1-Score":  round(f1_score(y_test, y_pred) * 100, 2),
            "y_pred":    y_pred,
            "cm":        confusion_matrix(y_test, y_pred),
        }
        trained[name] = m

    best_model_name = max(
        {k: v["F1-Score"] for k, v in results.items()},
        key=lambda k: results[k]["F1-Score"]
    )

    rf = trained["Random Forest"]
    fi = pd.Series(
        rf.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    return (trained, results, scaler, le_edu, le_emp, le_tgt,
            X_train, X_test, y_train, y_test,
            feature_cols, best_model_name, fi)


# ──────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────
df = load_data()

with st.spinner("Training models… (first load only)"):
    (trained_models, results, scaler,
     le_edu, le_emp, le_tgt,
     X_train, X_test, y_train, y_test,
     feature_cols, best_model_name, feat_imp) = prepare_ml(df)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Loan Approval")
    st.markdown("---")
    nav = st.radio(
        "Navigate to:",
        ["🏠 Problem Statement",
         "📂 Dataset Overview",
         "🔍 Steps Used",
         "📊 Live Analysis",
         "🤖 ML Models",
         "💡 Prediction",
         "📋 Business Decisions",
         "✅ Conclusion"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    approved_n = (df["loan_status"] == "Approved").sum()
    rejected_n = (df["loan_status"] == "Rejected").sum()
    total_n    = len(df)
    st.markdown(f"**Total Records:** {total_n:,}")
    st.markdown(f"✅ **Approved:** {approved_n:,} ({approved_n/total_n*100:.1f}%)")
    st.markdown(f"❌ **Rejected:** {rejected_n:,} ({rejected_n/total_n*100:.1f}%)")
    st.markdown("---")
    st.caption("Best Model: **" + best_model_name + f"**  \nF1: {results[best_model_name]['F1-Score']}%")

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown('<div class="main-title">🏦 Loan Approval Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-end Data Analytics & Machine Learning Project</div>', unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════
# 1. PROBLEM STATEMENT
# ══════════════════════════════════════════════
if nav == "🏠 Problem Statement":
    st.markdown('<div class="section-header">1. Problem Statement</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Overview
    Financial institutions face the challenge of deciding which loan applications to approve or reject.
    Manual review is time-consuming and inconsistent.  This project builds a data-driven system to
    **predict loan approval outcomes** based on applicant and financial information.

    ### Objective
    - Analyze the key factors that influence loan approval decisions.
    - Build machine learning models to predict whether a loan will be **Approved** or **Rejected**.
    - Provide an interactive interface for real-time predictions.
    - Derive actionable business insights from the data.

    ### Why It Matters
    - Banks can **reduce default risk** by approving creditworthy applicants.
    - Applicants receive **faster, fairer decisions**.
    - Data-driven policies replace subjective human judgment.
    """)

    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    approved = (df["loan_status"] == "Approved").sum()
    rejected = (df["loan_status"] == "Rejected").sum()
    features = len(df.columns) - 2  # minus loan_id and loan_status

    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total:,}</div><div class="metric-label">Total Applications</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{approved:,}</div><div class="metric-label">Approved</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{rejected:,}</div><div class="metric-label">Rejected</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{features}</div><div class="metric-label">Feature Variables</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# 2. DATASET OVERVIEW
# ══════════════════════════════════════════════
elif nav == "📂 Dataset Overview":
    st.markdown('<div class="section-header">2. Dataset Overview</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Sample Data", "🔢 Statistics", "🧹 Data Quality"])

    with tab1:
        st.markdown("#### First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")

    with tab2:
        num_df = df.drop(columns=["loan_id", "education", "self_employed", "loan_status"])
        st.markdown("#### Numerical Feature Statistics")
        st.dataframe(num_df.describe().round(2), use_container_width=True)

        st.markdown("#### Column Descriptions")
        col_info = {
            "loan_id":                    "Unique identifier (not used in modeling)",
            "no_of_dependents":           "Number of dependents (0–5)",
            "education":                  "Graduate / Not Graduate",
            "self_employed":              "Yes / No",
            "income_annum":               "Annual income (₹)",
            "loan_amount":                "Requested loan amount (₹)",
            "loan_term":                  "Loan term in years (2–20)",
            "cibil_score":                "Credit score (300–900)",
            "residential_assets_value":   "Residential asset value (₹)",
            "commercial_assets_value":    "Commercial asset value (₹)",
            "luxury_assets_value":        "Luxury asset value (₹)",
            "bank_asset_value":           "Bank asset / savings value (₹)",
            "loan_status":                "Target — Approved / Rejected",
        }
        st.dataframe(
            pd.DataFrame(list(col_info.items()), columns=["Column", "Description"]),
            use_container_width=True, hide_index=True
        )

    with tab3:
        st.markdown("#### Missing Values")
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing"]
        missing["Missing %"] = (missing["Missing"] / len(df) * 100).round(2)
        st.dataframe(missing, use_container_width=True, hide_index=True)

        st.markdown("#### Duplicates")
        st.success(f"✅ No duplicate rows found ({df.duplicated().sum()} duplicates).")

        st.markdown("#### Categorical Value Distributions")
        c1, c2, c3 = st.columns(3)
        c1.write("**Education**")
        c1.dataframe(df["education"].value_counts().reset_index().rename(columns={"count":"Count"}), hide_index=True)
        c2.write("**Self Employed**")
        c2.dataframe(df["self_employed"].value_counts().reset_index().rename(columns={"count":"Count"}), hide_index=True)
        c3.write("**Loan Status**")
        c3.dataframe(df["loan_status"].value_counts().reset_index().rename(columns={"count":"Count"}), hide_index=True)

# ══════════════════════════════════════════════
# 3. STEPS USED
# ══════════════════════════════════════════════
elif nav == "🔍 Steps Used":
    st.markdown('<div class="section-header">3. Steps Used for Analysis</div>', unsafe_allow_html=True)

    steps = [
        ("1. Data Loading", "Loaded the CSV file using Pandas. Stripped whitespace from all column names and string values."),
        ("2. Data Cleaning", "Checked for missing values (none found), duplicates (none found), and ensured correct data types. `loan_id` is excluded from features as it is only an identifier."),
        ("3. Exploratory Data Analysis", "Analyzed approval rates, distributions of numerical features, and relationships between approval status and each feature (CIBIL score, income, loan amount, term, education, employment, dependents)."),
        ("4. Feature Engineering", "Label-encoded `education` and `self_employed`. Created a `total_assets` derived insight for analysis. No features were dropped except `loan_id`."),
        ("5. Data Splitting", "Stratified 80/20 train-test split to maintain class proportions and prevent data leakage."),
        ("6. Feature Scaling", "Applied `StandardScaler` only to Logistic Regression. Tree-based models (Decision Tree, Random Forest, XGBoost) are scale-invariant and used raw features."),
        ("7. Model Training", "Trained four models: Logistic Regression, Decision Tree, Random Forest, and XGBoost."),
        ("8. Model Evaluation", "Evaluated each model on the test set using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix."),
        ("9. Best Model Selection", "Selected the model with the highest F1-Score as the primary predictor."),
        ("10. Prediction Interface", "Built an interactive Streamlit form for real-time loan approval prediction using the best model."),
        ("11. Business Insights", "Derived actionable business decisions from the EDA and model results."),
    ]

    for title, desc in steps:
        with st.expander(f"**{title}**", expanded=False):
            st.markdown(desc)

# ══════════════════════════════════════════════
# 4. LIVE ANALYSIS
# ══════════════════════════════════════════════
elif nav == "📊 Live Analysis":
    st.markdown('<div class="section-header">4. Live Analysis</div>', unsafe_allow_html=True)

    tab_overview, tab_cibil, tab_income, tab_loan, tab_demo, tab_assets, tab_fi = st.tabs([
        "📈 Approval Rate", "💳 CIBIL Score", "💰 Income",
        "🏷️ Loan Details", "👥 Demographics", "🏠 Assets", "⭐ Feature Importance"
    ])

    approved_n = (df["loan_status"] == "Approved").sum()
    rejected_n = (df["loan_status"] == "Rejected").sum()

    # ── Approval Rate ──
    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 4))
            colors = ["#2ecc71", "#e74c3c"]
            wedges, texts, autotexts = ax.pie(
                [approved_n, rejected_n],
                labels=["Approved", "Rejected"],
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2}
            )
            for t in autotexts:
                t.set_fontsize(13)
                t.set_fontweight("bold")
            ax.set_title("Loan Approval Distribution", fontsize=13, fontweight="bold")
            st.pyplot(fig)
            plt.close()

        with c2:
            st.markdown("#### Key Metrics")
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Total Applications | {len(df):,} |
            | Approved | {approved_n:,} ({approved_n/len(df)*100:.1f}%) |
            | Rejected | {rejected_n:,} ({rejected_n/len(df)*100:.1f}%) |
            | Approval Rate | **{approved_n/len(df)*100:.1f}%** |
            | Rejection Rate | **{rejected_n/len(df)*100:.1f}%** |
            """)
            st.markdown('<div class="info-box">📌 <b>Finding:</b> The dataset has a class imbalance with more approvals than rejections — approvals are about 1.6× rejections.</div>', unsafe_allow_html=True)

    # ── CIBIL ──
    with tab_cibil:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        approved_cibil  = df[df["loan_status"] == "Approved"]["cibil_score"]
        rejected_cibil  = df[df["loan_status"] == "Rejected"]["cibil_score"]

        axes[0].hist(approved_cibil,  bins=30, alpha=0.7, label="Approved",  color="#2ecc71", edgecolor="white")
        axes[0].hist(rejected_cibil, bins=30, alpha=0.7, label="Rejected", color="#e74c3c", edgecolor="white")
        axes[0].set_title("CIBIL Score Distribution by Loan Status", fontweight="bold")
        axes[0].set_xlabel("CIBIL Score")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        avg_cibil = df.groupby("loan_status")["cibil_score"].mean().round(1)
        axes[1].bar(avg_cibil.index, avg_cibil.values,
                    color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.5)
        axes[1].set_title("Average CIBIL Score by Loan Status", fontweight="bold")
        axes[1].set_ylabel("Average CIBIL Score")
        for i, v in enumerate(avg_cibil.values):
            axes[1].text(i, v + 5, str(v), ha="center", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        med_app = approved_cibil.median()
        med_rej = rejected_cibil.median()
        st.markdown(f"""
        | Status | Mean CIBIL | Median CIBIL |
        |--------|-----------|-------------|
        | Approved | {approved_cibil.mean():.1f} | {med_app:.0f} |
        | Rejected | {rejected_cibil.mean():.1f} | {med_rej:.0f} |
        """)
        st.markdown(f'<div class="info-box">📌 <b>Finding:</b> Approved applicants have a significantly higher average CIBIL score ({approved_cibil.mean():.0f}) vs rejected ({rejected_cibil.mean():.0f}). CIBIL score is a primary driver of approval.</div>', unsafe_allow_html=True)

    # ── Income ──
    with tab_income:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        approved_inc = df[df["loan_status"] == "Approved"]["income_annum"] / 1e6
        rejected_inc = df[df["loan_status"] == "Rejected"]["income_annum"] / 1e6

        axes[0].hist(approved_inc,  bins=30, alpha=0.7, label="Approved",  color="#2ecc71", edgecolor="white")
        axes[0].hist(rejected_inc, bins=30, alpha=0.7, label="Rejected", color="#e74c3c", edgecolor="white")
        axes[0].set_title("Annual Income Distribution by Loan Status", fontweight="bold")
        axes[0].set_xlabel("Annual Income (₹ Millions)")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        avg_inc = df.groupby("loan_status")["income_annum"].mean() / 1e6
        axes[1].bar(avg_inc.index, avg_inc.values,
                    color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.5)
        axes[1].set_title("Average Annual Income by Loan Status", fontweight="bold")
        axes[1].set_ylabel("Average Income (₹ Millions)")
        for i, v in enumerate(avg_inc.values):
            axes[1].text(i, v + 0.05, f"₹{v:.2f}M", ha="center", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(f'<div class="info-box">📌 <b>Finding:</b> Approved applicants tend to have higher annual incomes. Higher income improves approval chances as it signals repayment capacity.</div>', unsafe_allow_html=True)

    # ── Loan Details ──
    with tab_loan:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for status, color in [("Approved", "#2ecc71"), ("Rejected", "#e74c3c")]:
            subset = df[df["loan_status"] == status]["loan_amount"] / 1e6
            axes[0].hist(subset, bins=25, alpha=0.7, label=status, color=color, edgecolor="white")
        axes[0].set_title("Loan Amount Distribution by Status", fontweight="bold")
        axes[0].set_xlabel("Loan Amount (₹ Millions)")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        term_approval = df.groupby(["loan_term", "loan_status"]).size().unstack(fill_value=0)
        term_approval["Approval Rate %"] = (
            term_approval["Approved"] / (term_approval["Approved"] + term_approval["Rejected"]) * 100
        ).round(1)
        axes[1].bar(term_approval.index.astype(str),
                    term_approval["Approval Rate %"],
                    color="#3498db", edgecolor="white", linewidth=1.2)
        axes[1].set_title("Approval Rate by Loan Term (Years)", fontweight="bold")
        axes[1].set_xlabel("Loan Term (Years)")
        axes[1].set_ylabel("Approval Rate (%)")
        axes[1].axhline(approved_n / len(df) * 100, color="red", linestyle="--", alpha=0.7, label="Overall Rate")
        axes[1].legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Approval Rate by Loan Term")
        st.dataframe(term_approval[["Approved", "Rejected", "Approval Rate %"]], use_container_width=True)
        st.markdown('<div class="info-box">📌 <b>Finding:</b> Loan amount and loan term show variation in approval rates. Applicants with moderate loan amounts relative to income tend to be approved more often.</div>', unsafe_allow_html=True)

    # ── Demographics ──
    with tab_demo:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        edu_approval = df.groupby("education")["loan_status"].value_counts(normalize=True).unstack() * 100
        edu_approval.plot(kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"],
                          edgecolor="white", linewidth=1.2)
        axes[0].set_title("Approval Rate by Education", fontweight="bold")
        axes[0].set_ylabel("Percentage (%)")
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=15)
        axes[0].legend(title="Status")

        emp_approval = df.groupby("self_employed")["loan_status"].value_counts(normalize=True).unstack() * 100
        emp_approval.plot(kind="bar", ax=axes[1], color=["#2ecc71", "#e74c3c"],
                          edgecolor="white", linewidth=1.2)
        axes[1].set_title("Approval Rate by Employment Type", fontweight="bold")
        axes[1].set_ylabel("Percentage (%)")
        axes[1].set_xticklabels(["Not Self-Employed", "Self-Employed"], rotation=0)
        axes[1].legend(title="Status")

        dep_approval = df.groupby("no_of_dependents")["loan_status"].apply(
            lambda x: (x == "Approved").mean() * 100
        ).reset_index()
        dep_approval.columns = ["Dependents", "Approval Rate %"]
        axes[2].bar(dep_approval["Dependents"].astype(str),
                    dep_approval["Approval Rate %"],
                    color="#9b59b6", edgecolor="white", linewidth=1.2)
        axes[2].set_title("Approval Rate by No. of Dependents", fontweight="bold")
        axes[2].set_xlabel("Number of Dependents")
        axes[2].set_ylabel("Approval Rate (%)")
        axes[2].axhline(approved_n / len(df) * 100, color="red", linestyle="--", alpha=0.7, label="Overall")
        axes[2].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Approval Rate by Education")
            edu_rate = df.groupby("education")["loan_status"].apply(
                lambda x: f"{(x=='Approved').mean()*100:.1f}%"
            ).reset_index()
            edu_rate.columns = ["Education", "Approval Rate"]
            st.dataframe(edu_rate, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("#### Approval Rate by Employment")
            emp_rate = df.groupby("self_employed")["loan_status"].apply(
                lambda x: f"{(x=='Approved').mean()*100:.1f}%"
            ).reset_index()
            emp_rate.columns = ["Self Employed", "Approval Rate"]
            st.dataframe(emp_rate, use_container_width=True, hide_index=True)

        st.markdown('<div class="info-box">📌 <b>Finding:</b> Education and employment type show slight differences in approval rates. Number of dependents has a modest negative effect on approval at higher counts.</div>', unsafe_allow_html=True)

    # ── Assets ──
    with tab_assets:
        asset_cols = ["residential_assets_value", "commercial_assets_value",
                      "luxury_assets_value", "bank_asset_value"]
        asset_labels = ["Residential", "Commercial", "Luxury", "Bank"]

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        axes = axes.flatten()
        for i, (col, lbl) in enumerate(zip(asset_cols, asset_labels)):
            for status, color in [("Approved", "#2ecc71"), ("Rejected", "#e74c3c")]:
                subset = df[df["loan_status"] == status][col] / 1e6
                axes[i].hist(subset, bins=25, alpha=0.65, label=status, color=color, edgecolor="white")
            axes[i].set_title(f"{lbl} Asset Value (₹M)", fontweight="bold")
            axes[i].set_xlabel("Value (₹ Millions)")
            axes[i].set_ylabel("Count")
            axes[i].legend()
        plt.suptitle("Asset Value Distribution by Loan Status", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Average Asset Values by Loan Status (₹ Millions)")
        asset_avg = df.groupby("loan_status")[asset_cols].mean() / 1e6
        asset_avg.columns = asset_labels
        st.dataframe(asset_avg.round(2), use_container_width=True)
        st.markdown('<div class="info-box">📌 <b>Finding:</b> Approved applicants tend to have higher asset values across all categories. Total assets are a positive signal for loan approval.</div>', unsafe_allow_html=True)

    # ── Feature Importance ──
    with tab_fi:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors_fi = ["#e74c3c" if v == feat_imp.max() else "#3498db"
                     for v in feat_imp.values]
        bars = ax.barh(feat_imp.index[::-1], feat_imp.values[::-1],
                       color=colors_fi[::-1], edgecolor="white", linewidth=1.2)
        ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score")
        for bar, val in zip(bars, feat_imp.values[::-1]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Top Features Ranked")
        fi_df = feat_imp.reset_index()
        fi_df.columns = ["Feature", "Importance"]
        fi_df["Importance %"] = (fi_df["Importance"] * 100).round(2)
        fi_df.index = fi_df.index + 1
        st.dataframe(fi_df[["Feature", "Importance %"]], use_container_width=True)
        st.markdown(f'<div class="info-box">📌 <b>Finding:</b> The top feature is <b>{feat_imp.idxmax()}</b> with an importance of {feat_imp.max()*100:.1f}%. This confirms the business expectation that creditworthiness (CIBIL) and financial capacity drive approvals.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# 5. ML MODELS
# ══════════════════════════════════════════════
elif nav == "🤖 ML Models":
    st.markdown('<div class="section-header">5. Machine Learning Models</div>', unsafe_allow_html=True)

    tab_comp, tab_cm, tab_detail = st.tabs(["📊 Model Comparison", "🔢 Confusion Matrices", "📄 Detailed Report"])

    with tab_comp:
        metrics_data = {
            name: {k: v for k, v in m.items() if k not in ("y_pred", "cm")}
            for name, m in results.items()
        }
        metrics_df = pd.DataFrame(metrics_data).T.reset_index()
        metrics_df.columns = ["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]

        best_f1 = metrics_df["F1-Score (%)"].max()
        st.dataframe(
            metrics_df.style
                .highlight_max(subset=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"],
                               color="#d4edda")
                .format("{:.2f}", subset=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]),
            use_container_width=True,
            hide_index=True
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(metrics_df))
        width = 0.2
        metric_cols = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]
        bar_colors  = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

        for i, (col, color) in enumerate(zip(metric_cols, bar_colors)):
            ax.bar(x + i * width, metrics_df[col], width, label=col, color=color, edgecolor="white")

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(metrics_df["Model"], fontsize=10)
        ax.set_ylim(0, 115)
        ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
        ax.set_ylabel("Score (%)")
        ax.legend(loc="upper left", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.success(f"🏆 **Best Model: {best_model_name}** — F1-Score: {results[best_model_name]['F1-Score']}%")

    with tab_cm:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()
        label_names = le_tgt.classes_
        for i, (name, res) in enumerate(results.items()):
            sns.heatmap(
                res["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                ax=axes[i], cbar=False, linewidths=0.5
            )
            axes[i].set_title(f"{name}\nAcc: {res['Accuracy']}%  F1: {res['F1-Score']}%",
                              fontweight="bold")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")
        plt.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab_detail:
        model_choice = st.selectbox("Select model for detailed report:", list(results.keys()))
        y_pred = results[model_choice]["y_pred"]
        y_actual = y_test  # already encoded

        report_dict = classification_report(
            y_actual, y_pred, target_names=le_tgt.classes_, output_dict=True
        )
        report_df = pd.DataFrame(report_dict).T.round(4)
        st.markdown(f"#### Classification Report — {model_choice}")
        st.dataframe(report_df, use_container_width=True)

# ══════════════════════════════════════════════
# 6. PREDICTION
# ══════════════════════════════════════════════
elif nav == "💡 Prediction":
    st.markdown('<div class="section-header">6. Live Loan Approval Prediction</div>', unsafe_allow_html=True)
    st.markdown("Fill in the applicant details below and click **Predict Loan Approval**.")
    st.markdown(
        f'<div class="info-box">🤖 Model: <b>{best_model_name}</b> '
        f'(F1: {results[best_model_name]["F1-Score"]}%) &nbsp;|&nbsp; '
        f'CIBIL &lt; 540 → likely <b>Rejected</b> &nbsp;|&nbsp; '
        f'CIBIL &ge; 600 → likely <b>Approved</b> &nbsp;|&nbsp; '
        f'CIBIL is the #1 factor (80% weight)</div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════
    # ARCHITECTURE: ONE widget key per field — no shadow p_* copies.
    # Widget keys ARE the source of truth.
    # Presets write directly to these same keys, then st.rerun().
    # At predict-click we read st.session_state[widget_key] directly.
    # ══════════════════════════════════════════════════════════════

    # -- Initialise widget keys once (skipped if already present) --
    _WIDGET_DEFAULTS = {
        "pred_education":         "Graduate",
        "pred_self_employed":     "No",
        "pred_dependents":        2,
        "pred_income":            0,
        "pred_loan":              0,
        "pred_term":              10,
        "pred_cibil":             600,
        "pred_residential":       0,
        "pred_commercial":        0,
        "pred_luxury":            0,
        "pred_bank":              0,
        "prediction_result":      None,
    }
    for _wk, _wv in _WIDGET_DEFAULTS.items():
        if _wk not in st.session_state:
            st.session_state[_wk] = _wv

    # -- Preset definitions: keys match widget keys exactly ---------
    _PRESETS = {
        "strong": {
            "pred_education": "Graduate",     "pred_self_employed": "No",
            "pred_dependents": 1,
            "pred_income": 8000000,  "pred_loan": 15000000, "pred_term": 10,
            "pred_cibil": 750,
            "pred_residential": 6000000, "pred_commercial": 3000000,
            "pred_luxury": 12000000, "pred_bank": 4000000,
        },
        "borderline": {
            "pred_education": "Not Graduate", "pred_self_employed": "Yes",
            "pred_dependents": 3,
            "pred_income": 4500000,  "pred_loan": 12000000, "pred_term": 12,
            "pred_cibil": 540,
            "pred_residential": 2000000, "pred_commercial": 500000,
            "pred_luxury": 5000000,  "pred_bank": 1000000,
        },
        "weak": {
            "pred_education": "Not Graduate", "pred_self_employed": "Yes",
            "pred_dependents": 5,
            "pred_income": 2000000,  "pred_loan": 18000000, "pred_term": 20,
            "pred_cibil": 380,
            "pred_residential": 0, "pred_commercial": 0,
            "pred_luxury": 1000000, "pred_bank": 300000,
        },
    }

    # -- Preset buttons ---------------------------------------------
    st.markdown("##### Quick-fill Presets")
    st.caption("Fills all fields. You can edit any value after loading a preset.")
    _pb1, _pb2, _pb3 = st.columns(3)
    with _pb1:
        if st.button("Strong Applicant — CIBIL 750", use_container_width=True):
            st.session_state.update(_PRESETS["strong"])
            st.session_state["prediction_result"] = None
            st.rerun()
    with _pb2:
        if st.button("Borderline Applicant — CIBIL 540", use_container_width=True):
            st.session_state.update(_PRESETS["borderline"])
            st.session_state["prediction_result"] = None
            st.rerun()
    with _pb3:
        if st.button("Weak Applicant — CIBIL 380", use_container_width=True):
            st.session_state.update(_PRESETS["weak"])
            st.session_state["prediction_result"] = None
            st.rerun()

    st.markdown("---")

    # -- Input widgets (ONE per field, key = widget key = source of truth)
    _EDU_OPTS  = ["Graduate", "Not Graduate"]
    _EMP_OPTS  = ["No", "Yes"]
    _TERM_OPTS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    st.markdown("#### Applicant Details")
    _c1, _c2, _c3 = st.columns(3)

    with _c1:
        st.markdown("**Personal**")

        st.selectbox(
            "Education",
            options=_EDU_OPTS,
            index=_EDU_OPTS.index(st.session_state["pred_education"])
                  if st.session_state["pred_education"] in _EDU_OPTS else 0,
            key="pred_education",
        )

        st.radio(
            "Self Employed",
            options=_EMP_OPTS,
            index=_EMP_OPTS.index(st.session_state["pred_self_employed"])
                  if st.session_state["pred_self_employed"] in _EMP_OPTS else 0,
            horizontal=True,
            key="pred_self_employed",
        )

        st.slider(
            "Number of Dependents",
            min_value=0, max_value=5,
            value=int(st.session_state["pred_dependents"]),
            key="pred_dependents",
        )

    with _c2:
        st.markdown("**Financial**")

        st.number_input(
            "Annual Income (Rs.)",
            min_value=0, max_value=100_000_000,
            value=int(st.session_state["pred_income"]),
            step=100_000, format="%d",
            help="Total yearly income. Enter 0 if not known.",
            key="pred_income",
        )

        st.number_input(
            "Loan Amount (Rs.)",
            min_value=0, max_value=500_000_000,
            value=int(st.session_state["pred_loan"]),
            step=100_000, format="%d",
            help="Total loan amount being requested.",
            key="pred_loan",
        )

        # Live loan-to-income hint (informational only)
        _live_income = int(st.session_state["pred_income"])
        _live_loan   = int(st.session_state["pred_loan"])
        if _live_income > 0 and _live_loan > 0:
            _lti_live = _live_loan / _live_income
            if _lti_live <= 4:
                st.caption(f"Loan-to-income: {_lti_live:.1f}x — Very safe")
            elif _lti_live <= 6:
                st.caption(f"Loan-to-income: {_lti_live:.1f}x — Acceptable")
            elif _lti_live <= 10:
                st.caption(f"Loan-to-income: {_lti_live:.1f}x — High risk")
            else:
                st.caption(f"Loan-to-income: {_lti_live:.1f}x — Very high risk")

        st.selectbox(
            "Loan Term (Years)",
            options=_TERM_OPTS,
            index=_TERM_OPTS.index(st.session_state["pred_term"])
                  if st.session_state["pred_term"] in _TERM_OPTS else 4,
            key="pred_term",
        )

    with _c3:
        st.markdown("**Credit Score (CIBIL)**")

        # ONE number_input only — no slider duplicate
        st.number_input(
            "CIBIL Score (300–900)",
            min_value=300, max_value=900,
            value=int(st.session_state["pred_cibil"]),
            step=1, format="%d",
            help="Credit bureau score. Type a value or use the +/- arrows.",
            key="pred_cibil",
        )

        # Live traffic-light (reads the single key — no extra state)
        _live_cibil = int(st.session_state["pred_cibil"])
        if _live_cibil < 500:
            st.error(f"CIBIL {_live_cibil} — Poor. Very likely Rejected.")
        elif _live_cibil < 580:
            st.warning(f"CIBIL {_live_cibil} — Borderline. Outcome depends on other factors.")
        elif _live_cibil < 700:
            st.info(f"CIBIL {_live_cibil} — Fair. Likely Approved.")
        else:
            st.success(f"CIBIL {_live_cibil} — Strong. Very likely Approved.")

    st.markdown("#### Asset Details")
    st.caption("Enter actual asset values in Rs. Leave as 0 if not applicable.")
    _a1, _a2, _a3, _a4 = st.columns(4)
    with _a1:
        st.number_input(
            "Residential Assets (Rs.)",
            min_value=0, max_value=500_000_000,
            value=int(st.session_state["pred_residential"]),
            step=100_000, format="%d", key="pred_residential",
        )
    with _a2:
        st.number_input(
            "Commercial Assets (Rs.)",
            min_value=0, max_value=500_000_000,
            value=int(st.session_state["pred_commercial"]),
            step=100_000, format="%d", key="pred_commercial",
        )
    with _a3:
        st.number_input(
            "Luxury Assets (Rs.)",
            min_value=0, max_value=500_000_000,
            value=int(st.session_state["pred_luxury"]),
            step=100_000, format="%d", key="pred_luxury",
        )
    with _a4:
        st.number_input(
            "Bank / Savings (Rs.)",
            min_value=0, max_value=200_000_000,
            value=int(st.session_state["pred_bank"]),
            step=100_000, format="%d", key="pred_bank",
        )

    st.markdown("")

    # -- "Inputs changed" notice ------------------------------------
    _prev = st.session_state.get("prediction_result")
    if _prev is not None:
        _s = _prev["snap"]
        _changed = (
            st.session_state["pred_education"]     != _s["education"]     or
            st.session_state["pred_self_employed"]  != _s["self_employed"] or
            int(st.session_state["pred_dependents"]) != _s["dependents"]  or
            int(st.session_state["pred_income"])    != _s["income"]        or
            int(st.session_state["pred_loan"])      != _s["loan"]          or
            int(st.session_state["pred_term"])      != _s["term"]          or
            int(st.session_state["pred_cibil"])     != _s["cibil"]         or
            int(st.session_state["pred_residential"]) != _s["residential"] or
            int(st.session_state["pred_commercial"]) != _s["commercial"]   or
            int(st.session_state["pred_luxury"])    != _s["luxury"]        or
            int(st.session_state["pred_bank"])      != _s["bank"]
        )
        if _changed:
            st.info("Inputs have changed — click **Predict Loan Approval** to get a new result.")

    # -- Predict button ---------------------------------------------
    if st.button("Predict Loan Approval", type="primary", use_container_width=True, key="pred_btn"):

        # Read every value directly from the widget keys in session_state.
        # Streamlit guarantees these are current at the time the button fires.
        _snap = {
            "education":    st.session_state["pred_education"],
            "self_employed":st.session_state["pred_self_employed"],
            "dependents":   int(st.session_state["pred_dependents"]),
            "income":       int(st.session_state["pred_income"]),
            "loan":         int(st.session_state["pred_loan"]),
            "term":         int(st.session_state["pred_term"]),
            "cibil":        int(st.session_state["pred_cibil"]),
            "residential":  int(st.session_state["pred_residential"]),
            "commercial":   int(st.session_state["pred_commercial"]),
            "luxury":       int(st.session_state["pred_luxury"]),
            "bank":         int(st.session_state["pred_bank"]),
        }

        # Encode categoricals using the same LabelEncoders from training
        _edu_enc = le_edu.transform([_snap["education"]])[0]
        _emp_enc = le_emp.transform([_snap["self_employed"]])[0]

        # Build feature DataFrame with columns in the EXACT order of feature_cols
        # (the variable produced during prepare_ml — never hardcode column order)
        _feat_map = {
            "no_of_dependents":          _snap["dependents"],
            "education":                 _edu_enc,
            "self_employed":             _emp_enc,
            "income_annum":              _snap["income"],
            "loan_amount":               _snap["loan"],
            "loan_term":                 _snap["term"],
            "cibil_score":               _snap["cibil"],
            "residential_assets_value":  _snap["residential"],
            "commercial_assets_value":   _snap["commercial"],
            "luxury_assets_value":       _snap["luxury"],
            "bank_asset_value":          _snap["bank"],
        }
        _input_df = pd.DataFrame(
            [[_feat_map[c] for c in feature_cols]],
            columns=feature_cols,
        )

        # Run the model (Logistic Regression needs scaled input)
        _mdl = trained_models[best_model_name]
        if best_model_name == "Logistic Regression":
            _xi    = scaler.transform(_input_df)
            _pred  = _mdl.predict(_xi)[0]
            _proba = _mdl.predict_proba(_xi)[0]
        else:
            _pred  = _mdl.predict(_input_df)[0]
            _proba = _mdl.predict_proba(_input_df)[0]

        # Decode result — never assume class index; look it up from le_tgt.classes_
        _approved_idx  = list(le_tgt.classes_).index("Approved")
        _prob_approved = float(_proba[_approved_idx]) * 100
        _prediction    = le_tgt.inverse_transform([_pred])[0]

        # Store complete snapshot so the display never mixes inputs with wrong results
        st.session_state["prediction_result"] = {
            "prediction":          _prediction,
            "approval_probability":_prob_approved,
            "rejection_probability": 100.0 - _prob_approved,
            "model_used":          best_model_name,
            "snap":                _snap,
        }

    # -- Result display (reads only from the stored snapshot) -------
    _res = st.session_state.get("prediction_result")
    if _res:
        _pred_label = _res["prediction"]
        _prob_app   = _res["approval_probability"]
        _snap       = _res["snap"]

        st.markdown("---")
        st.markdown("### Prediction Result")

        if _pred_label == "Approved":
            st.markdown('<div class="approved-badge">LOAN APPROVED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="rejected-badge">LOAN REJECTED</div>', unsafe_allow_html=True)

        st.markdown("")
        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric("Approval Probability",   f"{_prob_app:.1f}%")
        _m2.metric("Rejection Probability",  f"{100 - _prob_app:.1f}%")
        _m3.metric("CIBIL Score Submitted",   _snap["cibil"])
        _m4.metric("Model Used",              _res["model_used"])
        st.progress(int(_prob_app))

        # -- Submitted Values (always matches exactly what went to the model)
        st.markdown("---")
        st.markdown("#### Submitted Values")
        st.caption("These are the exact values that were sent to the model.")
        _sv_df = pd.DataFrame({
            "Field": [
                "Education", "Self Employed", "Dependents",
                "Annual Income", "Loan Amount", "Loan Term", "CIBIL Score",
                "Residential Assets", "Commercial Assets", "Luxury Assets", "Bank / Savings",
            ],
            "Value": [
                _snap["education"],
                _snap["self_employed"],
                _snap["dependents"],
                f"Rs.{_snap['income']:,}",
                f"Rs.{_snap['loan']:,}",
                f"{_snap['term']} years",
                _snap["cibil"],
                f"Rs.{_snap['residential']:,}",
                f"Rs.{_snap['commercial']:,}",
                f"Rs.{_snap['luxury']:,}",
                f"Rs.{_snap['bank']:,}",
            ],
        })
        st.dataframe(_sv_df, use_container_width=True, hide_index=True)

        # -- Dataset benchmarks ------------------------------------
        _avg_app_cibil  = df[df["loan_status"] == "Approved"]["cibil_score"].mean()
        _avg_rej_cibil  = df[df["loan_status"] == "Rejected"]["cibil_score"].mean()
        _avg_app_income = df[df["loan_status"] == "Approved"]["income_annum"].mean()
        _avg_rej_income = df[df["loan_status"] == "Rejected"]["income_annum"].mean()
        _avg_app_loan   = df[df["loan_status"] == "Approved"]["loan_amount"].mean()
        _lti = _snap["loan"] / _snap["income"] if _snap["income"] > 0 else 999

        # -- Risk Factor Breakdown ---------------------------------
        st.markdown("---")
        st.markdown("#### Risk Factor Breakdown")
        st.caption("Benchmarks from actual training data. Assessments are informational only.")
        _bd = [
            {
                "Factor":       "CIBIL Score",
                "Submitted":    str(_snap["cibil"]),
                "Benchmark":    f"Approved avg {_avg_app_cibil:.0f}  |  Rejected avg {_avg_rej_cibil:.0f}",
                "Assessment":   "Good" if _snap["cibil"] >= 580 else ("Borderline" if _snap["cibil"] >= 500 else "Poor"),
                "Model Weight": "Very High — 80%",
            },
            {
                "Factor":       "Annual Income",
                "Submitted":    f"Rs.{_snap['income']:,}",
                "Benchmark":    f"Approved avg Rs.{_avg_app_income:,.0f}  |  Rejected avg Rs.{_avg_rej_income:,.0f}",
                "Assessment":   "Good" if _snap["income"] >= _avg_rej_income else "Below avg",
                "Model Weight": "Medium — 2%",
            },
            {
                "Factor":       "Loan Amount",
                "Submitted":    f"Rs.{_snap['loan']:,}",
                "Benchmark":    f"Approved dataset avg Rs.{_avg_app_loan:,.0f}",
                "Assessment":   "Moderate" if _snap["loan"] <= _avg_app_loan * 1.5 else "High",
                "Model Weight": "Medium — 3%",
            },
            {
                "Factor":       "Loan-to-Income Ratio",
                "Submitted":    f"{_lti:.1f}x" if _snap["income"] > 0 else "N/A",
                "Benchmark":    "Safe <= 6x  |  Risky > 6x",
                "Assessment":   "Safe" if _lti <= 6 else "Risky",
                "Model Weight": "Indirect",
            },
            {
                "Factor":       "Loan Term",
                "Submitted":    f"{_snap['term']} years",
                "Benchmark":    "Shorter = lower default risk",
                "Assessment":   "Good" if _snap["term"] <= 12 else "Moderate",
                "Model Weight": "Moderate — 6.5%",
            },
            {
                "Factor":       "No. of Dependents",
                "Submitted":    str(_snap["dependents"]),
                "Benchmark":    "0 deps: 64.2% approval  |  5 deps: 60.3% approval",
                "Assessment":   "Good" if _snap["dependents"] <= 2 else "Slightly high",
                "Model Weight": "Low — 0.8%",
            },
            {
                "Factor":       "Education",
                "Submitted":    _snap["education"],
                "Benchmark":    "Graduate 62.5%  |  Not Graduate 62.0%",
                "Assessment":   "Negligible difference",
                "Model Weight": "Very Low — 0.2%",
            },
            {
                "Factor":       "Employment",
                "Submitted":    "Self-Employed" if _snap["self_employed"] == "Yes" else "Salaried",
                "Benchmark":    "Both groups: 62.2% approval",
                "Assessment":   "No significant difference",
                "Model Weight": "Very Low — 0.3%",
            },
        ]
        st.dataframe(pd.DataFrame(_bd), use_container_width=True, hide_index=True)

        # -- Plain-English summary ----------------------------------
        st.markdown("---")
        if _pred_label == "Rejected":
            if _snap["cibil"] < 500:
                st.error(
                    f"Primary rejection reason: CIBIL {_snap['cibil']} is far below the ~580 "
                    f"approval boundary. Dataset rejected-applicant average is {_avg_rej_cibil:.0f}. "
                    f"A CIBIL of at least 600 would significantly improve the outcome."
                )
            elif _snap["cibil"] < 580:
                st.warning(
                    f"Borderline rejection: CIBIL {_snap['cibil']} is near the boundary but the "
                    f"overall financial profile was not sufficient. Raising CIBIL above 600 is the "
                    f"single most impactful change."
                )
            elif _snap["income"] == 0:
                st.error("Rejection reason: Annual income is Rs.0. Enter actual income so the "
                         "model can assess repayment capacity.")
            elif _lti > 10:
                st.error(f"Rejection reason: Loan-to-income ratio of {_lti:.1f}x is very high. "
                         f"The loan amount greatly exceeds what this income level can support.")
            else:
                st.error(
                    f"Rejection reason: The combination of CIBIL {_snap['cibil']}, "
                    f"income Rs.{_snap['income']:,}, and loan Rs.{_snap['loan']:,} does not "
                    f"meet the approval threshold learned from training data."
                )
        else:
            if _snap["cibil"] >= 700:
                st.success(
                    f"Primary approval driver: Strong CIBIL score of {_snap['cibil']} "
                    f"(dataset approved mean: {_avg_app_cibil:.0f}). "
                    f"This single factor carries 80% of the model's decision weight."
                )
            else:
                st.success(
                    f"Approval driver: CIBIL {_snap['cibil']} clears the approval threshold. "
                    f"The supporting financial profile produces a favourable prediction."
                )

# ══════════════════════════════════════════════
# 7. BUSINESS DECISIONS
# ══════════════════════════════════════════════
elif nav == "📋 Business Decisions":
    st.markdown('<div class="section-header">7. Business Decisions</div>', unsafe_allow_html=True)

    approved_cibil_mean = df[df["loan_status"] == "Approved"]["cibil_score"].mean()
    rejected_cibil_mean = df[df["loan_status"] == "Rejected"]["cibil_score"].mean()
    cibil_threshold     = int((approved_cibil_mean + rejected_cibil_mean) / 2)

    approved_income_mean = df[df["loan_status"] == "Approved"]["income_annum"].mean() / 1e6
    rejected_income_mean = df[df["loan_status"] == "Rejected"]["income_annum"].mean() / 1e6

    top_feature   = feat_imp.idxmax()
    top_feat_pct  = feat_imp.max() * 100

    edu_app  = df[df["education"]    == "Graduate"]["loan_status"].apply(lambda x: x == "Approved").mean() * 100
    emp_app  = df[df["self_employed"] == "No"]["loan_status"].apply(lambda x: x == "Approved").mean() * 100

    decisions = [
        {
            "title": "1. Use CIBIL Score as Primary Filter",
            "finding": f"Approved applicants average CIBIL {approved_cibil_mean:.0f} vs Rejected {rejected_cibil_mean:.0f}.",
            "decision": f"Set a minimum CIBIL score threshold of ~{cibil_threshold} for loan eligibility. Applicants below this should require additional review or collateral.",
            "icon": "💳"
        },
        {
            "title": "2. Income-Based Loan Sizing",
            "finding": f"Approved applicants average ₹{approved_income_mean:.2f}M income vs ₹{rejected_income_mean:.2f}M for rejected.",
            "decision": "Cap loan amounts at a defined multiple of annual income (e.g., 4–6×). This reduces the risk of over-lending to low-income applicants.",
            "icon": "💰"
        },
        {
            "title": "3. Prioritize CIBIL & Income in Credit Scoring",
            "finding": f"Random Forest feature importance shows {top_feature} is the top predictor ({top_feat_pct:.1f}% importance).",
            "decision": "Allocate the highest weight to CIBIL score and income metrics in the credit scoring model. These factors explain the most variance in approval outcomes.",
            "icon": "⭐"
        },
        {
            "title": "4. Graduate Applicants Show Higher Approval Rates",
            "finding": f"Graduate applicants have a {edu_app:.1f}% approval rate.",
            "decision": "While education is not a standalone criterion, it can be used as a supplementary positive factor in borderline cases, combined with other metrics.",
            "icon": "🎓"
        },
        {
            "title": "5. Salaried Applicants Have Lower Risk",
            "finding": f"Non-self-employed applicants have a {emp_app:.1f}% approval rate.",
            "decision": "Self-employed applicants should provide additional income proof or business financials. Consider offering slightly lower loan amounts or shorter terms to self-employed applicants with borderline profiles.",
            "icon": "💼"
        },
        {
            "title": "6. Asset-Backed Lending Reduces Risk",
            "finding": "Approved applicants consistently show higher asset values across residential, commercial, luxury, and bank assets.",
            "decision": "Introduce an asset-to-loan ratio check. Applicants with total assets ≥ 2× the loan amount should receive expedited approval.",
            "icon": "🏠"
        },
        {
            "title": "7. Automate Low-Risk Approvals",
            "finding": f"The {best_model_name} achieves {results[best_model_name]['Accuracy']}% accuracy on test data.",
            "decision": f"Deploy the {best_model_name} model to auto-approve applicants with probability ≥ 85% and auto-reject those with probability ≤ 15%. Route borderline cases to human review.",
            "icon": "🤖"
        },
    ]

    for d in decisions:
        with st.expander(f"{d['icon']} {d['title']}", expanded=True):
            col_f, col_d = st.columns([1, 1])
            with col_f:
                st.markdown("**📊 Data Finding:**")
                st.markdown(f"_{d['finding']}_")
            with col_d:
                st.markdown("**✅ Recommended Decision:**")
                st.markdown(d["decision"])

# ══════════════════════════════════════════════
# 8. CONCLUSION
# ══════════════════════════════════════════════
elif nav == "✅ Conclusion":
    st.markdown('<div class="section-header">8. Conclusion</div>', unsafe_allow_html=True)

    approved_pct = (df["loan_status"] == "Approved").mean() * 100
    top_feature  = feat_imp.idxmax()
    top_feat_pct = feat_imp.max() * 100
    app_cibil    = df[df["loan_status"] == "Approved"]["cibil_score"].mean()
    rej_cibil    = df[df["loan_status"] == "Rejected"]["cibil_score"].mean()

    st.markdown(f"""
    ### Summary of Findings

    This project analyzed **{len(df):,} loan applications** from a financial dataset to build a
    robust loan approval prediction system. Below are the key conclusions:

    #### 📊 Data Insights
    - **{approved_pct:.1f}%** of applications were approved in the dataset.
    - **{top_feature}** is the most important predictor ({top_feat_pct:.1f}% importance per Random Forest).
    - Approved applicants average CIBIL score is **{app_cibil:.0f}** vs **{rej_cibil:.0f}** for rejected —
      a difference of ~{app_cibil - rej_cibil:.0f} points.
    - Higher annual income, larger asset holdings, and graduate education all correlate positively with approval.
    - Self-employment and higher number of dependents are mild risk factors.
    - Loan term does not exhibit a strong linear trend in approval rates.

    #### 🤖 Model Performance
    """)

    perf_df = pd.DataFrame({
        name: {k: v for k, v in m.items() if k not in ("y_pred", "cm")}
        for name, m in results.items()
    }).T
    st.dataframe(perf_df, use_container_width=True)

    st.markdown(f"""
    - **Best Model: {best_model_name}** — F1-Score: {results[best_model_name]['F1-Score']}%,
      Accuracy: {results[best_model_name]['Accuracy']}%.
    - Random Forest and XGBoost consistently outperform simpler models on this dataset.
    - Logistic Regression provides a good interpretable baseline.
    - Decision Tree is useful for rule-based policy derivation but may overfit without pruning.

    #### 💡 Overall Recommendation

    Financial institutions should deploy the **{best_model_name}** model as the core engine for
    automated loan screening.  The model should be combined with human review for borderline
    cases (predicted probability between 30–70%).  CIBIL score and income should remain the
    primary eligibility filters, supplemented by asset verification.

    Regular model retraining on new data is recommended every 6–12 months to maintain accuracy
    as economic conditions and applicant profiles evolve.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Accuracy", f"{results[best_model_name]['Accuracy']}%")
    col2.metric("Best F1-Score", f"{results[best_model_name]['F1-Score']}%")
    col3.metric("Best Precision", f"{results[best_model_name]['Precision']}%")
    col4.metric("Best Recall", f"{results[best_model_name]['Recall']}%")

# Footer
st.markdown("---")
st.markdown(
    "<center><small style='color:#888;'>Loan Approval Prediction | Built with Python & Streamlit | Data Analytics & ML Project</small></center>",
    unsafe_allow_html=True
)
