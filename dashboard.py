# dashboard.py

# dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import glob
from io import BytesIO
import sys  # <-- 1. IMPORT SYS

# --- THIS IS THE FIX ---
# Add the project root directory (where this file is) to the Python path
# This allows it to find the 'src' module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
# ---------------------

# Now these local imports will work correctly
from src.customer_risk_analysis import compute_customer_risk, summarize_customer_risk
from src.transaction_report import export_transaction_pdf, generate_fraud_report
from src.visualize_clusters import generate_umap_projection, plot_clusters
from src.investigation_state import load_investigated, save_investigated
from src.visualization import plot_fraud_score_distribution, plot_top_customers_by_risk

# ... (rest of your code)

st.set_page_config(page_title="DeFraudify Dashboard", layout="wide")
sns.set(style="whitegrid")

def load_data(path):
    return pd.read_csv(path, parse_dates=["timestamp"])

def load_latest_model(pattern):
    models = sorted(glob.glob(pattern), reverse=True)
    if models:
        st.sidebar.success(f"Loaded: {os.path.basename(models[0])}")
        return joblib.load(models[0])
    st.sidebar.warning(f"No model found for {pattern}")
    return None

def load_feature_names():
    feature_path = "data/models/feature_names.txt"
    if os.path.exists(feature_path):
        with open(feature_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    st.sidebar.error("Feature names file not found. Please run supervised_pipeline.py first.")
    return None

rf_model = load_latest_model("data/models/random_forest_model_*.pkl")
xgb_model = load_latest_model("data/models/xgboost_model_*.pkl")
feature_names = load_feature_names()

st.title("💳 DeFraudify - Fraud Detection Dashboard")

with st.sidebar:
    st.header("Settings")
    data_path = st.text_input("Data file path", "data/processed/scored_transactions.csv")
    cluster_col = st.selectbox("Cluster column", ["kmeans_cluster", "dbscan_cluster"], index=0)
    st.markdown("---")
    if st.button("Reload Data"):
        st.experimental_rerun()

df = load_data(data_path)

tab_overview, tab_clusters, tab_customers, tab_reports, tab_predict = st.tabs(
    ["📊 Overview", "🗺️ Clusters", "👤 Customers", "📝 Reports", "🤖 Predict"]
)

with tab_overview:
    st.subheader("Fraud Score Distribution")
    st.pyplot(plot_fraud_score_distribution(df))

with tab_clusters:
    st.subheader("UMAP Clustering Visualization")
    features = ["amount", "hour", "day_of_week"]
    st.pyplot(generate_umap_projection(df, features, cluster_column=cluster_col))

    st.subheader("Cluster Scatterplot")
    st.pyplot(plot_clusters(df, cluster_col, x_feature="amount", y_feature="hour"))

with tab_customers:
    st.subheader("Customer-Level Risk Summary")
    customer_risk_df = compute_customer_risk(df)
    st.dataframe(summarize_customer_risk(customer_risk_df), use_container_width=True)

    st.subheader("Top Customers by Risk")
    st.pyplot(plot_top_customers_by_risk(customer_risk_df))

with tab_reports:
    st.subheader("Generate Fraud Report")
    threshold = st.slider("Fraud score threshold", 0.0, 1.0, 0.6, 0.01)
    report_df = generate_fraud_report(df, threshold=threshold)
    st.dataframe(report_df.head(50), use_container_width=True)

with tab_predict:
    st.subheader("Single Transaction Prediction")
    st.info("Enter transaction details to predict fraud probability:")

    amount = st.number_input("Amount", 0.0, 100.0)
    hour = st.slider("Transaction Hour", 0, 23, 12)
    day_of_week = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
    customer_avg_amount = st.number_input("Customer Avg Amount", 0.0, 200.0)
    customer_txn_count = st.number_input("Customer Transaction Count", 0, 10)
    isoforest_score = st.number_input("Isolation Forest Score", 0.0, 1.0, 0.5)
    lof_score = st.number_input("LOF Score", 0.0, 1.0, 0.5)

    input_features = {
        "amount": amount,
        "hour": hour,
        "day_of_week": day_of_week,
        "customer_avg_amount": customer_avg_amount,
        "customer_txn_count": customer_txn_count,
        "isoforest_score": isoforest_score,
        "lof_score": lof_score
    }

    input_df = pd.DataFrame([[input_features.get(f, 0) for f in feature_names]], columns=feature_names)

    if st.button("Predict Transaction"):
        if rf_model and xgb_model and feature_names:
            rf_prob = rf_model.predict_proba(input_df)[0, 1]
            xgb_prob = xgb_model.predict_proba(input_df)[0, 1]
            st.success(f"Random Forest Probability: {rf_prob:.4f}")
            st.success(f"XGBoost Probability: {xgb_prob:.4f}")
        else:
            st.error("Models or feature names not loaded.")

    st.markdown("---")
    st.subheader("Batch Prediction from CSV")
    batch_file = st.file_uploader("Upload CSV file", type=["csv"])

    if batch_file:
        batch_df = pd.read_csv(batch_file)
        st.write(batch_df.head())

        if feature_names and all(f in batch_df.columns for f in feature_names):
            if st.button("Predict Batch"):
                rf_probs = rf_model.predict_proba(batch_df[feature_names])[:, 1] if rf_model else None
                xgb_probs = xgb_model.predict_proba(batch_df[feature_names])[:, 1] if xgb_model else None

                batch_df["rf_probability"] = rf_probs
                batch_df["xgb_probability"] = xgb_probs

                def interpret(prob):
                    if prob < 0.3: return "Low"
                    if prob < 0.7: return "Medium"
                    return "High"

                batch_df["rf_risk_level"] = batch_df["rf_probability"].apply(interpret)
                batch_df["xgb_risk_level"] = batch_df["xgb_probability"].apply(interpret)

                st.success("Batch predictions completed.")
                st.dataframe(batch_df.head(50), use_container_width=True)

                csv = BytesIO()
                batch_df.to_csv(csv, index=False)
                csv.seek(0)

                st.download_button("Download CSV", data=csv, file_name="batch_predictions.csv", mime="text/csv")
        else:
            st.error(f"CSV must contain columns: {feature_names}")

st.sidebar.header("Investigation")
investigated = load_investigated()
txn_id = st.sidebar.text_input("Transaction ID to investigate")
if st.sidebar.button("Generate PDF Report"):
    txn = df[df["transaction_id"] == txn_id]
    if txn.empty:
        st.sidebar.error("Transaction not found.")
    else:
        reports_dir = os.path.join("data", "reports", "pdf")
        os.makedirs(reports_dir, exist_ok=True)
        path = os.path.join(reports_dir, f"{txn_id}_report.pdf")
        export_transaction_pdf(txn.iloc[0].to_dict(), path)
        st.sidebar.success(f"PDF saved to: {path}")
        investigated[txn_id] = "Investigated"
        save_investigated(investigated)

st.sidebar.markdown("---")
st.sidebar.caption("© DeFraudify 2024")
