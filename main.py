# main.py

from src.data_preprocessing import load_data, preprocess_data, save_processed_data
from src.clustering import apply_kmeans, apply_dbscan, apply_isolation_forest
from src.anomaly_scoring import score_isolation_forest, score_lof

import pandas as pd

if __name__ == "__main__":
    raw_path = "data/raw/sample_transactions.csv"
    processed_path = "data/processed/processed_transactions.csv"
    clustered_path = "data/processed/clustered_transactions.csv"
    scored_path = "data/processed/scored_transactions.csv"

    df = load_data(raw_path)
    if df is None:
        exit("[ERROR] Raw data not found.")

    df_processed = preprocess_data(df, missing_strategy="mean", encoding="onehot")

    for col in ["timestamp", "customer_id", "transaction_id", "is_fraud"]:
        if col in df.columns:
            df_processed[col] = df[col].values

    df_processed["hour"] = pd.to_datetime(df_processed["timestamp"]).dt.hour
    df_processed["day_of_week"] = pd.to_datetime(df_processed["timestamp"]).dt.dayofweek

    df_processed["amount"] = pd.to_numeric(df_processed["amount"], errors="coerce")

    if "customer_id" not in df_processed.columns or "amount" not in df_processed.columns:
        raise ValueError("Missing required columns: customer_id or amount")

    # Delete previous columns if they exist
    df_processed.drop(columns=["customer_avg_amount", "customer_txn_count"], errors="ignore", inplace=True)

    # Customer metrics calculation
    customer_avg = df_processed.groupby("customer_id")["amount"].mean().rename("customer_avg_amount")
    customer_txn_count = df_processed.groupby("customer_id").size().rename("customer_txn_count")

    customer_stats = pd.concat([customer_avg, customer_txn_count], axis=1).reset_index()

    df_processed = df_processed.merge(customer_stats, on="customer_id", how="left")

    save_processed_data(df_processed, processed_path)

    required_features = ["amount", "hour", "day_of_week", "customer_avg_amount", "customer_txn_count"]
    for col in required_features:
        if col not in df_processed.columns:
            print(df_processed.head())  # Visual debug in case something is missing
            raise ValueError(f"Required feature '{col}' missing from dataframe")

    X = df_processed[required_features].copy()

    df_processed["kmeans_cluster"] = apply_kmeans(X, n_clusters=5)
    df_processed["dbscan_cluster"] = apply_dbscan(X)
    df_processed["isolation_forest"] = apply_isolation_forest(X)

    df_processed.to_csv(clustered_path, index=False)
    print(f"[INFO] Clustered data saved to {clustered_path}")

    scores_if = score_isolation_forest(X)
    scores_lof = score_lof(X)

    for col in ["isoforest_score", "isoforest_flag", "lof_score", "lof_flag"]:
        if col in df_processed.columns:
            df_processed.drop(columns=col, inplace=True)

    final_df = pd.concat([df_processed.reset_index(drop=True), scores_if, scores_lof], axis=1)
    final_df["fraud_score"] = (final_df["isoforest_flag"] + final_df["lof_flag"]) / 2

    final_df.sort_values("isoforest_score", ascending=False, inplace=True)
    final_df.to_csv(scored_path, index=False)
    print(f"[INFO] Final scored transactions saved to {scored_path}")

