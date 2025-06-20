# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from joblib import dump

def create_pipeline():
    return Pipeline([
        ('feature_selection', SelectKBest(score_func=f_classif, k=8))  # Seleksi fitur otomatis
    ])

def preprocess_data(df, target_column, save_path):
    # Mengisi nilai kosong
    df['Customer Name'] = df['Customer Name'].fillna('Unknown')

    # Menghapus outlier
    numerical_features = df.select_dtypes(include='number').columns.to_list()
    for col in numerical_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # Drop kolom yang tidak diperlukan
    df = df.drop(columns=['Car_id', 'Date', 'Customer Name', 'Dealer_No', 'Phone'])

    # Rename kolom untuk konsistensi
    df.rename(columns={'Gender':'gender', 'Annual Income':'annual_income_$', 'Dealer_Name':'dealer_name',
                       'Company':'company', 'Model':'model', 'Engine':'engine', 'Transmission':'transmission',
                       'Color':'color', 'Price ($)':'price_$', 'Body Style':'body_style',
                       'Dealer_Region':'dealer_region'}, inplace=True)

    # Mengelompokkan data numerik menjadi kategori
    df['annual_income_category'] = pd.qcut(df['annual_income_$'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    df['price_category'] = pd.qcut(df['price_$'], q=3, labels=['Cheap', 'Affordable', 'Expensive'])

    # Encode kolom kategorikal
    categorical_columns = ['gender', 'dealer_name', 'company', 'engine', 'model',
                           'transmission', 'color', 'body_style', 'dealer_region',
                           'annual_income_category', 'price_category']

    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Simpan label encoders
    dump(label_encoders, save_path)

    # Pisahkan fitur dan target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Buat pipeline dan terapkan pada data
    pipeline = create_pipeline()
    X_transformed = pipeline.fit_transform(X, y)

    # Membagi dataset menjadi data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, pipeline, label_encoders

# Contoh penggunaan:
#X_train, X_test, y_train, y_test, pipeline, label_encoders = preprocess_data(car_df, 'annual_income_category', 'label_encoders.joblib')