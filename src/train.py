import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


def load_data(path: str):
    df = pd.read_csv(path)

    X = df[
        [
            "age",
            "income",
            "loan_amount",
            "loan_duration_months",
            "credit_score",
            "previous_defaults",
        ]
    ]
    y = df["approved"]

    return X, y


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])


def main():
    print("Training pipeline started\n")

    # Load data
    X, y = load_data("data/credit_data.csv")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = build_pipeline()
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="recall"
    )

    print("\nCross Validation Recall scores:")
    print(cv_scores)
    print(f"Mean CV Recall: {cv_scores.mean():.2f}")

    # ===== Default threshold (0.5) =====
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=== Evaluation (threshold = 0.5) ===")
    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # ===== Custom threshold (0.7) =====
    probs = model.predict_proba(X_test)[:, 1]
    custom_predictions = (probs >= 0.7).astype(int)

    print("\n=== Evaluation (threshold = 0.7) ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, custom_predictions))
    cm = confusion_matrix(y_test, custom_predictions)

    tn, fp, fn, tp = cm.ravel()

    print("\nDecision metrics (threshold = 0.7):")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")

    print("Classification Report:")
    print(classification_report(y_test, custom_predictions))

    # Debug / inspection
    print("Predicted probabilities:")
    print(probs)

    # Save model
    joblib.dump(model, "model.joblib")
    print("\nModel saved as model.joblib")
if __name__ == "__main__":
    main()
