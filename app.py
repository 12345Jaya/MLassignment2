import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

with open(MODEL_DIR / "metadata.json") as f:
    metadata = json.load(f)

st.set_page_config(page_title="ML Model Comparison", layout="wide")
st.title("Machine Learning Classification Model Comparison")

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}

model_name = st.selectbox("Select a trained model", list(model_files))
uploaded_file = st.file_uploader("Upload labelled test CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    target = metadata["target_column"]
    features = metadata["feature_columns"]
    missing = [c for c in features + [target] if c not in data.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        X_test = data[features].apply(pd.to_numeric)
        y_test = data[target].astype(str)
        model = joblib.load(MODEL_DIR / model_files[model_name])
        predictions = model.predict(X_test)

        try:
            probabilities = model.predict_proba(X_test)
            auc = roc_auc_score(
                y_test, probabilities, labels=model.classes_,
                multi_class="ovr", average="weighted"
            )
        except (ValueError, AttributeError):
            auc = float("nan")

        metrics = {
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": auc,
            "Precision": precision_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "F1": f1_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "MCC": matthews_corrcoef(y_test, predictions)
        }

        st.subheader("Evaluation metrics")
        cols = st.columns(6)
        for col, (label, value) in zip(cols, metrics.items()):
            col.metric(label, f"{value:.4f}" if pd.notna(value) else "N/A")

        st.subheader("Confusion matrix")
        labels = sorted(set(y_test) | set(predictions))
        cm = confusion_matrix(y_test, predictions, labels=labels)
        fig, ax = plt.subplots(figsize=(9, 7))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax, xticks_rotation=45, colorbar=False
        )
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Classification report")
        report = classification_report(
            y_test, predictions, zero_division=0, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose())

        output = data.copy()
        output["Predicted_Class"] = predictions
        st.download_button(
            "Download predictions",
            output.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )
