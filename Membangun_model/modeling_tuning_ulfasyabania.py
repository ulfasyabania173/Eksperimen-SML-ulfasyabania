import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# Load preprocessed data
DATA_PATH = 'Preprocessing/iris_preprocessing.csv'
df = pd.read_csv(DATA_PATH)

# Pisahkan fitur dan target
X = df.iloc[:, :-1]
y = df['target']

# Split ulang untuk memastikan reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning grid
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 200]
}

mlflow.set_experiment('Iris-Classification-Tuning-ulfasyabania')
best_acc = 0
best_params = None

for C in param_grid['C']:
    for solver in param_grid['solver']:
        for max_iter in param_grid['max_iter']:
            with mlflow.start_run(run_name=f'LogReg_C{C}_solver{solver}_iter{max_iter}'):
                # Inisialisasi model dengan hyperparameter
                model = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)

                # Manual logging
                mlflow.log_param('C', C)
                mlflow.log_param('solver', solver)
                mlflow.log_param('max_iter', max_iter)
                mlflow.log_param('model_type', 'LogisticRegression')
                mlflow.log_metric('accuracy', float(acc))
                mlflow.sklearn.log_model(model, 'model')

                # Log classification report sebagai artefak
                import json
                with open('classification_report.json', 'w') as f:
                    json.dump(report, f)
                mlflow.log_artifact('classification_report.json')

                print(f"C={C}, solver={solver}, max_iter={max_iter}, Akurasi: {acc}")
                if acc > best_acc:
                    best_acc = acc
                    best_params = {'C': C, 'solver': solver, 'max_iter': max_iter}

print(f"Best accuracy: {best_acc} with params: {best_params}")
print("Model dan artefak telah disimpan di MLflow Tracking UI.")
print("Buka MLflow UI dengan perintah: mlflow ui")
