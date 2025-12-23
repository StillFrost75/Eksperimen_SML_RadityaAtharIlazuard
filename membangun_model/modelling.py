import pandas as pd
import sys
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==============================================================================
# 1. SETUP PATH & LOAD DATA
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
# 2. EXPERIMENT SETUP
# ==============================================================================
mlflow.set_experiment("Eksperimen_Heart_Disease_Basic")

# ==============================================================================
# 3. TRAINING DENGAN PURE AUTOLOG (Syarat Basic)
# ==============================================================================
# Kita aktifkan autolog TANPA parameter log_models=False.
# Biarkan dia default (True), jadi dia otomatis simpan model, param, metric.
mlflow.sklearn.autolog()

print("Mulai training model basic (Pure Autolog)...")

with mlflow.start_run(run_name="Basic_Training_Autolog_Only", nested=True):
     mlflow.autolog() ):
    
    # Inisialisasi Model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Proses Training (Fit)
    # Di sinilah MAGIC autolog bekerja. Dia akan otomatis mencatat:
    # - Parameters (n_estimators, random_state, dll)
    # - Metrics (Training score)
    # - Artifacts (Model.pkl, conda.yaml, MLmodel)
    rf.fit(X_train, y_train)
    
    # Evaluasi (Opsional, hanya untuk print di terminal)
    # Autolog biasanya otomatis menghitung metrics training, 
    # tapi untuk test metrics kadang perlu manual atau eval_and_log_metrics.
    # Namun untuk syarat 'Basic', kode di atas sudah cukup memicu autolog.
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc:.4f}")
    
    print("Selesai. Semua log ditangani otomatis oleh MLflow Autolog.")