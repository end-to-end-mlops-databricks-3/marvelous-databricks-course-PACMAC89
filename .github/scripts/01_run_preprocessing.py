from src.mlops_course import preprocessing

def main():
    # Daten laden
    train_df, _ = preprocessing.load_data()

    # Aufteilen
    X_train, X_test, _, _ = preprocessing.split_data(train_df)

    # Speichern
    preprocessing.save_data(X_train, X_test, "processed")

    print("✅ Preprocessing completed successfully.")

if __name__ == "__main__":
    main()