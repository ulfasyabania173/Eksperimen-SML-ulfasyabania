# Eksperimen-MSML-ulfasyabania

## Deskripsi
Eksperimen ini menggunakan dataset Iris untuk demonstrasi proses machine learning, mulai dari pengambilan data, preprocessing otomatis, modeling, hyperparameter tuning, hingga automasi dengan GitHub Actions dan integrasi MLflow ke DagsHub.

## Struktur Proyek
- `Preprocessing/automate_ulfasyabania.py`: Script preprocessing otomatis
- `Preprocessing/iris_raw.csv`: Dataset mentah hasil ekstraksi dari scikit-learn
- `Preprocessing/iris_preprocessing.csv`: Dataset hasil preprocessing
- `Membangun_model/modeling_ulfasyabania.py`: Training model Logistic Regression + MLflow autolog (lokal)
- `Membangun_model/modeling_tuning_ulfasyabania.py`: Hyperparameter tuning + manual logging MLflow (lokal)
- `Membangun_model/modeling_tuning_dagshub_ulfasyabania.py`: Hyperparameter tuning + manual logging MLflow ke DagsHub (online)
- `Membangun_model/requirements.txt`: Daftar dependensi Python untuk modeling
- `.github/workflows/preprocess.yml`: Workflow GitHub Actions untuk preprocessing otomatis
- `Eksperimen_MSML_ulfasyabania.txt`: Draf laporan proses keseluruhan
- `DagsHub.txt`: Laporan eksperimen MLflow di DagsHub

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
   python Membangun_model/modeling_ulfasyabania.py
   ```
3. Jalankan hyperparameter tuning (MLflow lokal):
   ```bash
   python Membangun_model/modeling_tuning_ulfasyabania.py
   ```
4. Jalankan hyperparameter tuning & logging ke DagsHub:
   ```bash
   python Membangun_model/modeling_tuning_dagshub_ulfasyabania.py
   ```

## Automasi di GitHub Actions
Setiap ada perubahan pada script atau dataset, workflow akan otomatis menjalankan preprocessing dan mengunggah hasilnya sebagai artifact.

## Tautan Repository
- GitHub: https://github.com/ulfasyabania173/Eksperimen-SML-ulfasyabania
- DagsHub Experiments: https://dagshub.com/ulfasyabania173/Eksperimen-SML-ulfasyabania/experiments
- DagsHub MLflow UI: https://dagshub.com/ulfasyabania173/Eksperimen-SML-ulfasyabania.mlflow

## Catatan
- Semua proses dapat direplikasi dari repo ini.
- Untuk tracking online, pastikan token DagsHub dan konfigurasi MLflow sudah benar.
- Lihat file Eksperimen_MSML_ulfasyabania.txt dan DagsHub.txt untuk laporan detail.
