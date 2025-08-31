# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    model = joblib.load("fraud_model.pkl")
    return model


model = load_model()

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="💳 Credit Card Fraud Detector",
    page_icon="💰",
    layout="wide"
)

# ==========================
# Sidebar
# ==========================
st.sidebar.title("⚙️ Options")
st.sidebar.markdown("""
This app detects **fraudulent credit card transactions** using a trained XGBoost model.

**You can either:**
1. Upload a CSV file with transactions
2. Input a single transaction manually
""")

option = st.sidebar.radio("Choose Input Method:", ["Upload CSV", "Manual Input"])

# ==========================
# CSV Upload Section
# ==========================
if option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(data.head())

        if st.button("Predict Fraud 🚨"):
            predictions = model.predict(data)
            probabilities = model.predict_proba(data)[:, 1]
            data["Prediction"] = predictions
            data["Fraud Probability"] = probabilities


            # Color-coded display
            def color_pred(val):
                if val == 1:
                    color = 'background-color: #ff9999'
                else:
                    color = 'background-color: #99ff99'
                return color


            st.subheader("Prediction Results")
            st.dataframe(data.style.applymap(color_pred, subset=["Prediction"]))

            # Summary metrics
            st.subheader("Summary Metrics")
            total = len(data)
            fraud_count = data["Prediction"].sum()
            non_fraud = total - fraud_count

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", total)
            col2.metric("Fraudulent", fraud_count, delta=f"{(fraud_count / total) * 100:.2f}%")
            col3.metric("Non-Fraudulent", non_fraud, delta=f"{(non_fraud / total) * 100:.2f}%")

            # Probability distribution
            st.subheader("Fraud Probability Distribution")
            fig, ax = plt.subplots()
            sns.histplot(probabilities, bins=20, kde=True, color='skyblue', ax=ax)
            ax.set_xlabel("Fraud Probability")
            ax.set_ylabel("Count")
            st.pyplot(fig)

    else:
        st.info("Please upload a CSV file to make predictions.")

# ==========================
# Manual Input Section
# ==========================
elif option == "Manual Input":
    st.subheader("Enter Transaction Details Manually")

    # Example features: Time, V1-V28, Amount
    # For brevity, we'll use Amount and 3 sample V-features; you can expand to all 28
    Time = st.number_input("Time", min_value=0, value=0)
    V1 = st.number_input("V1", value=0.0)
    V2 = st.number_input("V2", value=0.0)
    V3 = st.number_input("V3", value=0.0)
    Amount = st.number_input("Amount", min_value=0.0, value=0.0)

    if st.button("Predict Single Transaction 🚨"):
        input_data = pd.DataFrame([[Time, V1, V2, V3, Amount]],
                                  columns=["Time", "V1", "V2", "V3", "Amount"])
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0, 1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"🚨 Fraudulent Transaction! Probability: {prob:.2f}")
        else:
            st.success(f"✅ Non-Fraudulent Transaction. Probability: {prob:.2f}")
