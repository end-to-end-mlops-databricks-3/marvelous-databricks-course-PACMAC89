import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#import matplotlib.pyplot as plt
#import seaborn as sns
from pathlib import Path

# Dynamisch berechneter Datenpfad (unabhängig vom Ausführungsort)
def get_data_path(subdir="raw"):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../data/{subdir}"))

def load_data():
    data_path = get_data_path("raw")
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    print(f"Train Data Shape: {train_df.shape}")
    print(f"Test Data Shape: {test_df.shape}")
    return train_df, test_df

def split_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_data(X_train, X_test):
    # Features bestimmen
    numerical_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X_train.select_dtypes(include=["object"]).columns

    # Pipelines definieren
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # ColumnTransformer kombinieren
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Fit/Transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed

def save_data(train_df, test_df, output_path):
    output_path = Path(output_path)  # 🔧 Sicherstellen, dass es ein Path-Objekt ist
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_path / "train.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

# Optional: Visualisierungen (auskommentiert für Headless-Umgebungen)
# def plot_missing_values(df):
#     plt.figure(figsize=(12, 6))
#     sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
#     plt.title("Missing Values")
#     plt.show()

# def plot_target_distribution(y):
#     plt.figure(figsize=(10, 6))
#     sns.histplot(y, kde=True)
#     plt.title("Distribution of SalePrice")
#     plt.show()