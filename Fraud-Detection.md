# Intelligent Analysis of Anomalous Transactions

This is a complete solution for detecting anomalous financial transactions, identifying customer-level risk, and applying supervised machine learning for fraud prediction. The system combines unsupervised anomaly detection, clustering, and supervised models to flag potentially fraudulent activity.

---

## Project Overview

**1. Data Preprocessing:**  
Cleans raw transaction data, extracts time-based features, encodes categorical variables, and normalizes amounts.

**2.Behavioral Features:**  
Calculates customer-level statistics like average transaction amount and transaction frequency.

**3. Unsupervised Detection:**  
Applies KMeans and DBSCAN clustering, Isolation Forest, and Local Outlier Factor (LOF) to detect anomalies without labeled data.

**4. Supervised Machine Learning (Optional):**  
Trains Random Forest and XGBoost models using labeled transactions to predict fraud probability.

**5. Dashboard (Streamlit UI):**  
Interactive web dashboard to visualize clusters, risk scores, customer profiles, generate PDF reports, and run real-time or batch fraud predictions.

## Project Structure

```plaintext
DeFraudify/
├── data/
│   ├── raw/                   # Raw transaction data
│   ├── processed/             # Processed datasets with features and scores
│   ├── reports/               # Generated PDF reports and evaluation metrics
│   └── models/                # Trained ML models (.pkl files)
├── scripts/
│   └── generate_sample_data.py   # Script to generate synthetic transaction data
├── src/
│   ├── data_preprocessing.py     # Data cleaning and feature engineering
│   ├── clustering.py             # KMeans and DBSCAN clustering
│   ├── anomaly_scoring.py        # Isolation Forest & LOF scoring
│   ├── customer_risk_analysis.py # Aggregates fraud risk per customer
│   ├── transaction_report.py     # PDF report generation for flagged transactions
│   ├── visualize_clusters.py     # UMAP & cluster visualizations
│   ├── visualization.py          # Fraud score and risk visualizations
│   ├── investigation_state.py    # State management for investigated transactions
├── main.py                       # Main pipeline for unsupervised detection
├── supervised_pipeline.py        # Pipeline for supervised model training & evaluation
├── dashboard.py                  # Streamlit dashboard for interactive analysis
└── README.md                      # Project documentation
```

# How It Works

## Step 1: Data Processing

```plaintext
python main.py
```

- Loads raw transactions
- Cleans and preprocesses data
- Generates customer behavior features
- Applies clustering and anomaly detection
- Outputs scored dataset

## Step 2 (Optional): Supervised Model Training

```plaintext
python supervised_pipeline.py
```

- Trains Random Forest & XGBoost on labeled transactions
- Saves models and evaluation reports
- Alerts if model performance is below acceptable thresholds

## Step 3: Launch the Dashboard

```plaintext
streamlit run dashboard.py
```

or

```plaintext
python -m streamlit run dashboard.py
```

Features:
- Fraud score distribution
- UMAP cluster visualizations
- Customer risk rankings
- Fraud report generation (PDF)
- Real-time or batch transaction prediction


## Effectiveness & Limitations
| Component | Description                    | Current Effectiveness                                                                                 |
|------|---------------------------------|-------------------------------------------------------------------------------------------|
| Unsupervised Detection    | Clustering + Isolation Forest + LOF                | ~65-75% anomaly detection accuracy (on synthetic data)  |
| Supervised ML Models    | Random Forest, XGBoost (optional) | Up to ~85% accuracy with labeled, realistic datasets                      |

**Note:** Current synthetic data does not fully reflect real-world fraud patterns. For production-level performance, integration with real, labeled datasets is required.

## Requirements

- Python 3.9+
- pandas, scikit-learn, seaborn, matplotlib
- xgboost, imbalanced-learn
- umap-learn, fpdf, streamlit

Install dependencies:
```plaintext
pip install -r requirements.txt
```
---

## Future Improvements

- Enhance synthetic data realism
- Add advanced feature engineering
- Hyperparameter tuning for models
- Expand dashboard functionality
