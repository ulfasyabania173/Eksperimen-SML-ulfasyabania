# Eksperimen-MSML-ulfasyabania

## Deskripsi
Eksperimen ini menggunakan dataset Iris untuk demonstrasi proses machine learning, mulai dari pengambilan data, preprocessing otomatis, hingga automasi dengan GitHub Actions.

## Struktur Proyek
- `Preprocessing/automate_ulfasyabania.py`: Script preprocessing otomatis
- `Preprocessing/iris_raw.csv`: Dataset mentah hasil ekstraksi dari scikit-learn
- `Preprocessing/iris_preprocessing.csv`: Dataset hasil preprocessing
- `.github/workflows/preprocess.yml`: Workflow GitHub Actions untuk preprocessing otomatis
- `Eksperimen_MSML_ulfasyabania.txt`: Draf laporan proses

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

## Automasi di GitHub Actions
Setiap ada perubahan pada script atau dataset, workflow akan otomatis menjalankan preprocessing dan mengunggah hasilnya sebagai artifact.

## Tautan Repository
https://github.com/ulfasyabania173/Eksperimen-SML-ulfasyabania

## Catatan
