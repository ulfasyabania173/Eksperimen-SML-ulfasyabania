import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_iris(input_csv='iris_raw.csv', output_csv='iris_preprocessing.csv', test_size=0.2, random_state=42):
    """
    Melakukan preprocessing pada dataset Iris:
    - Menghapus duplikasi
    - Memisahkan fitur dan target
    - Standarisasi fitur
    - Split data menjadi train dan test
    - Menyimpan hasil preprocessing ke file CSV baru
    """
    # Load data
    df = pd.read_csv(input_csv)

    # Hapus duplikasi
    df = df.drop_duplicates()

    # Pisahkan fitur dan target
    X = df.iloc[:, :-1]
    y = df['target']

    # Standarisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    # Gabungkan kembali fitur dan target untuk data training dan testing
    train_df = pd.DataFrame(X_train, columns=X.columns)
    train_df['target'] = y_train.values
    test_df = pd.DataFrame(X_test, columns=X.columns)
    test_df['target'] = y_test.values

    # Gabungkan train dan test untuk disimpan
    processed_df = pd.concat([train_df, test_df], ignore_index=True)
    processed_df.to_csv(output_csv, index=False)
    return processed_df

if __name__ == "__main__":
    preprocess_iris()
    print("Preprocessing selesai. File iris_preprocessing.csv telah disimpan.")
