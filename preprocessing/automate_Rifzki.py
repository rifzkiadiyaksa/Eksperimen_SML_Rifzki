import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def preprocess_data(input_path, output_dir):
    """
    Fungsi untuk memuat, memproses, dan menyimpan data.
    """
    # Pastikan direktori output ada (dalam kasus ini, direktori saat ini)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Memuat Data
    try:
        # Path disesuaikan untuk membaca dari root direktori proyek
        full_input_path = os.path.join('..', input_path)
        df = pd.read_csv(full_input_path)
        print(f"Data berhasil dimuat dari {full_input_path}")
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {full_input_path}")
        return

    # 2. Encoding Data Kategorikal
    df['GENDER'] = df['GENDER'].replace({'M': 1, 'F': 0})
    df['LUNG_CANCER'] = df['LUNG_CANCER'].replace({'YES': 1, 'NO': 0})
    print("Encoding data kategorikal selesai.")

    # 3. Memisahkan Fitur dan Target
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']

    # 4. Pembagian Data (Splitting)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Pembagian data menjadi train dan test set selesai.")

    # 5. Normalisasi Fitur
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Mengubah kembali ke DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    print("Normalisasi data selesai.")

    # 6. Menyimpan Hasil Preprocessing
    train_data = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)
    
    # Path output disesuaikan untuk menyimpan di direktori saat ini (preprocessing/)
    train_output_path = os.path.join(output_dir, 'lung_cancer_train_preprocessed.csv')
    test_output_path = os.path.join(output_dir, 'lung_cancer_test_preprocessed.csv')
    scaler_output_path = os.path.join(output_dir, 'scaler.joblib')

    train_data.to_csv(train_output_path, index=False)
    test_data.to_csv(test_output_path, index=False)
    joblib.dump(scaler, scaler_output_path)
    
    print(f"Data yang telah diproses disimpan di direktori: {output_dir}")

if __name__ == '__main__':
    # File raw data berada di root, satu level di atas skrip ini
    input_file_name = 'survey_lung_cancer_raw.csv'
    # Direktori output adalah direktori saat ini tempat skrip dijalankan
    output_directory = '.' 
    
    preprocess_data(input_file_name, output_directory)