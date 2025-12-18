import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data(path: str):
    df = pd.read_csv(path)
    X = df[["age", "study_hours"]]
    y = df["exam_score"]
    return X, y


def build_pipeline():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])
    return pipeline


def main():
    print("Training pipeline started")

    X, y = load_data("data/credit_data.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print(f"Model accuracy: {acc:.2f}")


if __name__ == "__main__":
    main()
