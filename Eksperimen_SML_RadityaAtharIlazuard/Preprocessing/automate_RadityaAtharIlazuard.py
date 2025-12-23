import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(path):
    """
    Fungsi untuk membaca data dari file CSV
    """
    try:
        df = pd.read_csv(path)
        print(f"Data berhasil dimuat dari {path}")
        return df
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {path}")
        return None

def preprocess_data(df):
    """
    Fungsi utama untuk cleaning dan preprocessing data.
    Mengembalikan: X_train, X_test, y_train, y_test, preprocessor
    """
    # 1. Cleaning Awal
    df_clean = df.copy()
    
    # Ubah target 'num' jadi binary
    df_clean['num'] = df_clean['num'].apply(lambda x: 1 if x > 0 else 0)
    
    # Buang kolom kotor
    cols_to_drop = ['id', 'dataset', 'ca', 'thal', 'slope']
    df_clean = df_clean.drop(columns=cols_to_drop)
    
    # 2. Splitting Data
    X = df_clean.drop(columns=['num'])
    y = df_clean['num']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Setup Pipeline
    numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 4. Fit & Transform
    # Kita fit pada X_train, lalu transform X_train dan X_test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

if __name__ == "__main__":
    import os
    
    # 1. Cari tahu di mana file script ini (automate_....py) berada
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Buat path menuju dataset relatif terhadap lokasi script ini
    # Artinya: Dari folder script ini, MUNDUR satu langkah (..), lalu masuk folder 'heart_disease_uci_raw', lalu cari 'heart.csv'
    # Pastikan nama folder 'heart_disease_uci_raw' SAMA PERSIS dengan nama folder di kiri layar VS Code Anda.
    data_path = os.path.join(current_script_dir, '..', 'heart_disease_uci_raw', 'heart.csv')
    
    # (Opsional) Print path-nya biar yakin Python nyari di mana
    print(f"Mencoba mencari file di: {os.path.normpath(data_path)}")
    
    df = load_data(data_path)
    
    if df is not None:
        X_train, X_test, y_train, y_test, _ = preprocess_data(df)
        print("Test Run Berhasil!")
        print(f"Shape X_train: {X_train.shape}")