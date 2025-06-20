import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# Load preprocessed data
DATA_PATH = '../Preprocessing/iris_preprocessing.csv'
df = pd.read_csv(DATA_PATH)

# Pisahkan fitur dan target
X = df.iloc[:, :-1]
y = df['target']

# Split ulang untuk memastikan reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mulai MLflow run
mlflow.set_experiment('Iris-Classification-ulfasyabania')
with mlflow.start_run(run_name='LogisticRegression'):
    # Inisialisasi dan latih model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Log parameter, metrik, dan model ke MLflow
    mlflow.log_param('model_type', 'LogisticRegression')
    mlflow.log_metric('accuracy', float(acc))
    mlflow.sklearn.log_model(model, 'model')

    # Log classification report sebagai artefak
    import json
    with open('classification_report.json', 'w') as f:
        json.dump(report, f)
    mlflow.log_artifact('classification_report.json')

    print(f"Akurasi: {acc}")
    print("Model dan artefak telah disimpan di MLflow Tracking UI.")
    print("Buka MLflow UI dengan perintah: mlflow ui")
