
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Load Model and Files
# -----------------------------
from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("insurance_fraud_model.json")
label_encoders = joblib.load("label_encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Insurance Claim Fraud Detection System")

st.write(
    "Predict whether an insurance claim is fraudulent using Machine Learning."
)

st.sidebar.header("Enter Claim Details")

input_data = {}

# -----------------------------
# User Inputs
# -----------------------------
for feature in feature_names:

    if feature in label_encoders:

        options = list(label_encoders[feature].classes_)

        value = st.sidebar.selectbox(
            feature,
            options
        )

        encoded = label_encoders[feature].transform([value])[0]

        input_data[feature] = encoded

    else:

        value = st.sidebar.number_input(
            feature,
            value=0.0
        )

        input_data[feature] = value

input_df = pd.DataFrame([input_data])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("🚨 Fraudulent Claim")

    else:

        st.success("✅ Genuine Claim")

    st.metric(
        "Fraud Probability",
        f"{probability[1]*100:.2f}%"
    )

    st.metric(
        "Non-Fraud Probability",
        f"{probability[0]*100:.2f}%"
    )

    st.progress(int(probability[1]*100))

    st.subheader("Top Feature Importance")

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    st.dataframe(importance.head(10))

    # -----------------------------
    # SHAP
    # -----------------------------
    st.subheader("SHAP Explanation")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots(figsize=(8,6))

    shap.summary_plot(
        shap_values,
        input_df,
        show=False
    )

    st.pyplot(fig)
