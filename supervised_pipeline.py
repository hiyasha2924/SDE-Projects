# supervised_pipeline.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from fpdf import FPDF
import os
import joblib
from datetime import datetime

ALERT_THRESHOLD = 0.75
ALERT_LOG_PATH = "data/reports/alerts_log.txt"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

df = pd.read_csv("data/processed/scored_transactions.csv")

if 'is_fraud' not in df.columns:
    raise ValueError("Dataset must contain 'is_fraud' column with true fraud labels (0 or 1)")

features = [
    "amount", "isoforest_score", "lof_score",
    "hour", "day_of_week",
    "customer_avg_amount", "customer_txn_count"
]

X = df[features].copy()
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)

# Save feature names for consistent predictions
os.makedirs("data/models", exist_ok=True)
with open("data/models/feature_names.txt", "w") as f:
    f.write("\n".join(features))

# Save models
joblib.dump(rf_model, f"data/models/random_forest_model_{timestamp}.pkl")
joblib.dump(xgb_model, f"data/models/xgboost_model_{timestamp}.pkl")

# Evaluate
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

os.makedirs("data/reports", exist_ok=True)

new_metrics = pd.DataFrame({
    "timestamp": [timestamp],
    "roc_auc_rf": [roc_auc_score(y_test, y_prob_rf)],
    "roc_auc_xgb": [roc_auc_score(y_test, y_prob_xgb)]
})

history_path = "data/reports/model_history.csv"
if os.path.exists(history_path):
    pd.concat([pd.read_csv(history_path), new_metrics]).to_csv(history_path, index=False)
else:
    new_metrics.to_csv(history_path, index=False)

with open(f"data/reports/random_forest_report_{timestamp}.txt", "w") as f:
    f.write(classification_report(y_test, y_pred_rf))
with open(f"data/reports/xgboost_report_{timestamp}.txt", "w") as f:
    f.write(classification_report(y_test, y_pred_xgb))

# Visualizations
sns.set(style="whitegrid")

plt.figure(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_test, y_prob_rf, name="Random Forest", ax=plt.gca())
RocCurveDisplay.from_predictions(y_test, y_prob_xgb, name="XGBoost", ax=plt.gca())
plt.title("ROC Curves - Random Forest vs. XGBoost")
roc_path = f"data/reports/model_roc_curves_{timestamp}.png"
plt.tight_layout()
plt.savefig(roc_path)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Random Forest Confusion Matrix")
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("XGBoost Confusion Matrix")
conf_path = f"data/reports/confusion_matrices_{timestamp}.png"
plt.tight_layout()
plt.savefig(conf_path)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
importances_rf = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
sns.barplot(x=importances_rf.values, y=importances_rf.index, ax=axes[0], color="skyblue")
axes[0].set_title("Random Forest Feature Importance")

importances_xgb = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=False)
sns.barplot(x=importances_xgb.values, y=importances_xgb.index, ax=axes[1], color="lightgreen")
axes[1].set_title("XGBoost Feature Importance")
importance_path = f"data/reports/feature_importances_{timestamp}.png"
plt.tight_layout()
plt.savefig(importance_path)
plt.close()

# Performance history plot
if os.path.exists(history_path):
    df_hist = pd.read_csv(history_path)
    plt.figure(figsize=(8, 6))
    plt.plot(df_hist["timestamp"], df_hist["roc_auc_rf"], marker="o", label="Random Forest AUC")
    plt.plot(df_hist["timestamp"], df_hist["roc_auc_xgb"], marker="s", label="XGBoost AUC")
    plt.xticks(rotation=45)
    plt.ylabel("ROC AUC Score")
    plt.title("Model ROC AUC Over Time")
    plt.legend()
    history_graph_path = f"data/reports/performance_history_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(history_graph_path)
    plt.close()
else:
    history_graph_path = None

# PDF report
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Fraud Detection Model Report", ln=True, align="C")

pdf.set_font("Arial", "", 12)
pdf.ln(5)
pdf.cell(0, 10, f"Random Forest ROC AUC: {roc_auc_score(y_test, y_prob_rf):.4f}", ln=True)
pdf.cell(0, 10, f"XGBoost ROC AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}", ln=True)

pdf.ln(10)
pdf.cell(0, 10, "ROC Curves:", ln=True)
pdf.image(roc_path, w=180)

pdf.ln(5)
pdf.cell(0, 10, "Confusion Matrices:", ln=True)
pdf.image(conf_path, w=180)

pdf.ln(5)
pdf.cell(0, 10, "Feature Importances:", ln=True)
pdf.image(importance_path, w=180)

if history_graph_path:
    pdf.ln(5)
    pdf.cell(0, 10, "Performance History:", ln=True)
    pdf.image(history_graph_path, w=180)

report_path = f"data/reports/model_evaluation_report_{timestamp}.pdf"
pdf.output(report_path)

print(f"[INFO] Reports saved to data/reports with timestamp {timestamp}")
print(f"[INFO] Models saved to data/models with timestamp {timestamp}")

# Alert handling
alert_triggered = False
if roc_auc_score(y_test, y_prob_rf) < ALERT_THRESHOLD or roc_auc_score(y_test, y_prob_xgb) < ALERT_THRESHOLD:
    alert_triggered = True
    print("\n[ALERT] Model performance below acceptable threshold! Please review.")
    with open(ALERT_LOG_PATH, "a") as log:
        log.write(f"[{timestamp}] ALERT: AUC below {ALERT_THRESHOLD}. RF AUC: {roc_auc_score(y_test, y_prob_rf):.4f}, XGB AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}\n")
else:
    with open(ALERT_LOG_PATH, "a") as log:
        log.write(f"[{timestamp}] OK: Model performance within acceptable range. RF AUC: {roc_auc_score(y_test, y_prob_rf):.4f}, XGB AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}\n")
