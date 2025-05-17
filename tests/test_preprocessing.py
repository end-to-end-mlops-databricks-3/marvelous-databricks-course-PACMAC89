# tests/test_preprocessing.py
from pathlib import Path  # Falls nicht schon vorhanden
import os
import pandas as pd
import pytest
from house_price import data_processor

def test_load_data() -> None:
    """Testet das Laden von Trainings- und Testdaten."""
    train_df, test_df = data_processor.load_data()
    assert isinstance(train_df, pd.DataFrame), "Train data is not a DataFrame"
    assert isinstance(test_df, pd.DataFrame), "Test data is not a DataFrame"
    assert not train_df.empty, "Train data is empty"
    assert not test_df.empty, "Test data is empty"

def test_split_data_shapes() -> None:
    """Testet die Form der gesplitteten Daten."""
    df, _ = data_processor.load_data()
    X_train, X_test, y_train, y_test = data_processor.split_data(df)
    assert X_train.shape[0] == y_train.shape[0], "Mismatch in X_train and y_train rows"
    assert X_test.shape[0] == y_test.shape[0], "Mismatch in X_test and y_test rows"

def test_save_data_creates_files(tmp_path: Path) -> None:
    """Testet, ob die save_data-Funktion Dateien erzeugt."""
    # Dummy data
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    output_dir = tmp_path / "processed"
    
    data_processor.save_data(df, df, output_dir)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    assert train_path.exists(), "train.csv was not created"
    assert test_path.exists(), "test.csv was not created"

    # Optional: Check file contents
    loaded = pd.read_csv(train_path)
    assert loaded.equals(df), "Saved and loaded train.csv do not match"
