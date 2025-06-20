# Eksperimen-MSML-ulfasyabania

## Deskripsi
Eksperimen ini menggunakan dataset Iris untuk demonstrasi proses machine learning, mulai dari pengambilan data, preprocessing otomatis, modeling, hyperparameter tuning, hingga automasi dengan GitHub Actions dan integrasi MLflow ke DagsHub. Pipeline ini juga mendukung monitoring model dengan Prometheus dan visualisasi di Grafana.

## Struktur Proyek
- `Preprocessing/automate_ulfasyabania.py`: Script preprocessing otomatis
- `Preprocessing/iris_raw.csv`: Dataset mentah hasil ekstraksi dari scikit-learn
- `Preprocessing/iris_preprocessing.csv`: Dataset hasil preprocessing
- `Membangun_model/modeling_ulfasyabania.py`: Training model Logistic Regression + MLflow autolog (lokal)
- `Membangun_model/modeling_tuning_ulfasyabania.py`: Hyperparameter tuning + manual logging MLflow (lokal)
- `Membangun_model/modeling_tuning_dagshub_ulfasyabania.py`: Hyperparameter tuning + manual logging MLflow ke DagsHub (online)
- `Membangun_model/requirements.txt`: Daftar dependensi Python untuk modeling
- `Workflow-CI/MLProject/MLproject`: Definisi pipeline MLflow Project
- `Workflow-CI/MLProject/conda.yaml`: Environment pipeline
- `.github/workflows/preprocess.yml`: Workflow GitHub Actions untuk preprocessing otomatis
- `.github/workflows/ci-mlflow.yml`: Workflow CI/CD MLflow Project
- `Eksperimen_MSML_ulfasyabania.txt`: Draf laporan proses keseluruhan
- `DagsHub.txt`: Laporan eksperimen MLflow di DagsHub
- `Monitoring/`, `Logging/`: Folder setup monitoring Prometheus, Grafana, dan logging

## Cara Menjalankan Preprocessing Secara Lokal
1. Pastikan Python dan pip sudah terinstal.
2. Install dependency:
   ```bash
   pip install pandas scikit-learn
   ```
3. Jalankan script:
   ```bash
   python Preprocessing/automate_ulfasyabania.py
   ```

## Cara Menjalankan Modeling & Tuning
1. Install semua dependensi modeling:
   ```bash
   pip install -r Membangun_model/requirements.txt
   ```
2. Jalankan training model dasar (MLflow lokal):
   ```bash
   python Membangun_model/modeling_ulfasyabania.py Preprocessing/iris_preprocessing.csv
   ```
3. Jalankan hyperparameter tuning (MLflow lokal):
   ```bash
   python Membangun_model/modeling_tuning_ulfasyabania.py
   ```
4. Jalankan hyperparameter tuning & logging ke DagsHub:
   ```bash
   python Membangun_model/modeling_tuning_dagshub_ulfasyabania.py
   ```

## Cara Menjalankan Pipeline MLflow Project
1. Masuk ke folder utama project.
2. Jalankan:
   ```bash
   mlflow run Workflow-CI/MLProject -P data_path=Preprocessing/iris_preprocessing.csv
   ```

## Automasi di GitHub Actions
Setiap ada perubahan pada script atau dataset, workflow akan otomatis menjalankan preprocessing dan pipeline MLflow Project, serta mengunggah hasilnya sebagai artifact ke halaman Actions GitHub.

## Monitoring & Logging
- Model yang di-serve dengan MLflow dapat diekspos ke Prometheus pada endpoint `/metrics`.
- Prometheus dapat di-setup untuk scrape metrik model MLflow (lihat folder Monitoring).
- Visualisasi dan alerting dapat dilakukan dengan Grafana.
- Contoh query Prometheus dan setup dashboard Grafana tersedia di dokumentasi.

## Tautan Repository
- GitHub: https://github.com/ulfasyabania173/Eksperimen-SML-ulfasyabania
- DagsHub Experiments: https://dagshub.com/ulfasyabania173/Eksperimen-SML-ulfasyabania/experiments
- DagsHub MLflow UI: https://dagshub.com/ulfasyabania173/Eksperimen-SML-ulfasyabania.mlflow

## Catatan
- Semua proses dapat direplikasi dari repo ini.
- Untuk tracking online, pastikan token DagsHub dan konfigurasi MLflow sudah benar.
- Lihat file Eksperimen_MSML_ulfasyabania.txt, DagsHub.txt, dan Workflow-CI.txt untuk laporan dan instruksi detail.
- Untuk monitoring Prometheus, pastikan model di-log dengan MLflow >=2.4.0 dan dependency `prometheus-client` sudah ada di environment model.
- Jika ingin serve model dan monitoring, jalankan:
  ```bash
  mlflow models serve -m <path_model> -p 1234 --env-manager=conda
  ```
  lalu lakukan request prediksi ke endpoint `/invocations` dan cek `/metrics`.
