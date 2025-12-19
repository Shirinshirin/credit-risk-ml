import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


# -----------------------
# Data
# -----------------------
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


# -----------------------
# Model
# -----------------------
def build_pipeline():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ]
    )


# -----------------------
# Main
# -----------------------
def main():
    print("Training pipeline started\n")

    # Load & split data
    X, y = load_data("data/credit_data.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build model
    model = build_pipeline()

    # Cross Validation (on train only)
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="f1"
    )

    print("Cross Validation F1 scores:")
    print(cv_scores)
    print(f"Mean CV F1: {cv_scores.mean():.2f}\n")

    # Train final model
    model.fit(X_train, y_train)

    # Predict probabilities
    probs = model.predict_proba(X_test)[:, 1]
    # -----------------------
    # ROC & AUC
    # -----------------------
    auc = roc_auc_score(y_test, probs)
    fpr, tpr, thresholds = roc_curve(y_test, probs)

    print(f"\nROC AUC Score: {auc:.2f}")

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

    # -----------------------
    # Default threshold (0.5)
    # -----------------------
    y_pred_default = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_default)

    print("=== Evaluation (threshold = 0.5) ===")
    print(f"Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred_default))

    # -----------------------
    # Threshold analysis
    # -----------------------
    thresholds = [0.3, 0.5, 0.7]

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

        print(f"\nThreshold = {t}")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    # -----------------------
    # Final decision threshold (0.7)
    # -----------------------
    final_threshold = 0.7
    final_preds = (probs >= final_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, final_preds).ravel()

    print(f"\n=== Final Decision Metrics (threshold = {final_threshold}) ===")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")
    print(classification_report(y_test, final_preds))

    # -----------------------
    # Save model
    # -----------------------
    joblib.dump(model, "model.joblib")
    print("\nModel saved as model.joblib")


if __name__ == "__main__":
    main()
