import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="ML Classification App", layout="centered")

st.title("🩺 Breast Cancer Classification Demo")
st.write("""
Upload your **test CSV file**, choose a model, and see predictions plus evaluation metrics.
(Your file must contain a column named **'target'**.)
""")

# -----------------------------
# MODEL SELECTION
# -----------------------------
model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic_Regression",
        "Decision_Tree",
        "KNN",
        "Naive_Bayes",
        "Random_Forest",
        "XGBoost"
    ]
)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload test data (CSV only)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Basic check
    if "target" not in df.columns:
        st.error("Your CSV must contain a column named 'target'.")
        st.stop()

    X = df.drop("target", axis=1)
    y = df["target"]

    # Load selected model
    model_path = f"model/{model_choice}.pkl"

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)

    # Predictions
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    st.subheader("📊 Evaluation Metrics")

    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    mcc = matthews_corrcoef(y, preds)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(acc, 4))
    col2.metric("AUC", round(auc, 4))
    col3.metric("Precision", round(prec, 4))

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", round(rec, 4))
    col5.metric("F1 Score", round(f1, 4))
    col6.metric("MCC", round(mcc, 4))

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    st.subheader("📉 Confusion Matrix")

    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # -----------------------------
    # CLASSIFICATION REPORT
    # -----------------------------
    st.subheader("📄 Classification Report")
    st.text(classification_report(y, preds))

    # Show sample predictions
    st.subheader("🔍 Sample Predictions")
    result_df = df.copy()
    result_df["prediction"] = preds
    st.dataframe(result_df.head(20))
