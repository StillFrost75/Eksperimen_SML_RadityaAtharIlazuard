import pandas as pd
import sys
import os
import json
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# === PERBAIKAN ERROR "main thread is not in main loop" ===
import matplotlib
matplotlib.use('Agg') # Wajib ditaruh SEBELUM import pyplot
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. SETUP PATH & IMPORT DATA
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
preprocessing_dir = os.path.join(current_dir, '..', 'preprocessing')
sys.path.append(preprocessing_dir)

try:
    from automate_RadityaAtharIlazuard import load_data, preprocess_data
except ImportError:
    sys.exit("Gagal import automate script.")

data_path = os.path.join(current_dir, '..', 'heart_disease_uci_raw', 'heart.csv')
df = load_data(data_path)
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

# ==============================================================================
# 2. HYPERPARAMETER TUNING
# ==============================================================================
mlflow.set_experiment("Eksperimen_Heart_Disease_Tuning")
# Autolog untuk mencatat proses tuning, tapi model final kita log manual


print("Mulai Grid Search...")
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# ==============================================================================
# 3. MANUAL LOGGING
# ==============================================================================
print("Menyimpan hasil Manual Logging...")

# Tambahkan nested=True agar aman dijalankan via GitHub Actions
with mlflow.start_run(run_name="Manual_Logging_Best_Model", nested=True):
    mlflow.autolog() 
    # A. LOG PARAMETERS
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    
    # B. HITUNG METRICS
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    # C. ARTIFACTS TAMBAHAN (JSON & GAMBAR)
    metrics_dict = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
    json_filename = "metric_info.json"
    with open(json_filename, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    mlflow.log_artifact(json_filename) 
    
    # Karena kita sudah pakai matplotlib.use('Agg'), kode di bawah aman
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    cm_filename = "training_confusion_matrix.png"
    plt.savefig(cm_filename)
    # plt.close() penting untuk membersihkan memori
    plt.close() 
    mlflow.log_artifact(cm_filename)
    
    # D. LOG MODEL DENGAN SIGNATURE & INPUT EXAMPLE
    print("Menyimpan model dengan signature...")
    signature = infer_signature(X_train, y_pred)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature,
        input_example=X_train[:5], 
    )
    
    print("Logging Selesai. Struktur folder MLflow sudah lengkap.")

    # Bersihkan file sementara
    if os.path.exists(json_filename): os.remove(json_filename)
    if os.path.exists(cm_filename): os.remove(cm_filename)