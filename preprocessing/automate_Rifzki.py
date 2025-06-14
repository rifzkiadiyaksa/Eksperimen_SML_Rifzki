import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys
import os

def preprocess_and_save(input_filename, output_filename):
    """
    Memuat data mentah, melakukan encoding dan normalisasi, 
    lalu menyimpannya sebagai satu file bersih.
    """

    if not input_filename or not output_filename:
        print("Error: Nama file input dan output harus diberikan.")
        sys.exit(1)

    try:
        df = pd.read_csv(input_filename)
        print(f"Data berhasil dimuat dari '{input_filename}'")
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di '{input_filename}'")
        sys.exit(1)

    # --- Proses Preprocessing ---

    # 1. Encoding Data Kategorikal
    df['GENDER'] = df['GENDER'].replace({'M': 1, 'F': 0})
    df['LUNG_CANCER'] = df['LUNG_CANCER'].replace({'YES': 1, 'NO': 0})
    print("Encoding data kategorikal selesai.")

    # 2. Normalisasi Fitur
    # Memisahkan fitur dari target sebelum scaling
    features = df.drop('LUNG_CANCER', axis=1)
    target = df['LUNG_CANCER']
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Menggabungkan kembali fitur yang sudah di-scale dengan target
    df_clean = pd.DataFrame(scaled_features, columns=features.columns)
    df_clean['LUNG_CANCER'] = target.values
    print("Normalisasi data selesai.")
    
    # --- Penyimpanan Hasil ---

    # Memastikan direktori output ada
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Menyimpan dataset yang sudah bersih
    df_clean.to_csv(output_filename, index=False)
    print(f"Dataset bersih telah disimpan di '{output_filename}'")

    # Menyimpan scaler untuk digunakan nanti
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler telah disimpan di '{scaler_path}'")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        preprocess_and_save(input_file, output_file)
    else:
        print("Penggunaan: python automate_Rifzki.py <nama_file_input> <nama_file_output>")
        sys.exit(1)