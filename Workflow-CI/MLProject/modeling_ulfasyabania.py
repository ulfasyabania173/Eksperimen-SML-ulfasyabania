import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import sys
import prometheus_client

# requirements tambahan untuk monitoring Prometheus
# Pastikan prometheus-client terinstall
# Untuk conda: conda install -c conda-forge prometheus_client
# Untuk pip: pip install prometheus-client

# Aktifkan autolog MLflow
mlflow.sklearn.autolog()

# Load preprocessed data dari argument command line
DATA_PATH = sys.argv[1]
df = pd.read_csv(DATA_PATH)

# Pisahkan fitur dan target
X = df.iloc[:, :-1]
y = df['target']

# Split ulang untuk memastikan reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mulai MLflow run
mlflow.set_experiment('Iris-Classification-ulfasyabania')
with mlflow.start_run(run_name='LogisticRegression') as run:
    # Inisialisasi dan latih model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Log classification report sebagai artefak
    import json
    with open('classification_report.json', 'w') as f:
        json.dump(report, f)
    mlflow.log_artifact('classification_report.json')

    print(f"Akurasi: {acc}")
    print("Model dan artefak telah disimpan di MLflow Tracking UI.")
    print("Buka MLflow UI dengan perintah: mlflow ui")
    # Print path model MLflow
    model_uri = f"mlruns/{run.info.experiment_id}/{run.info.run_id}/artifacts/model"
    print(f"Path model MLflow: {model_uri}")
