# # app.py
# import streamlit as st
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # ==========================
# # Load Model
# # ==========================
# @st.cache_resource
# def load_model():
#     model_path = r"C:\Users\HP\Desktop\movi\fraud_model.pkl"
#     model = joblib.load(model_path)
#     return model
#
# model = load_model()
#
# # ==========================
# # Page Configuration
# # ==========================
# st.set_page_config(
#     page_title="💳 Credit Card Fraud Detector",
#     page_icon="💰",
#     layout="wide"
# )
#
# # ==========================
# # Sidebar
# # ==========================
# st.sidebar.title("⚙️ Options")
# st.sidebar.markdown("""
# This app detects **fraudulent credit card transactions** using a trained XGBoost model.
#
# **You can either:**
# 1. Upload a CSV file with transactions
# 2. Input a single transaction manually
# """)
#
# option = st.sidebar.radio("Choose Input Method:", ["Upload CSV", "Manual Input"])
#
# # ==========================
# # Expected Features
# # ==========================
# FEATURES = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9',
#             'V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
#             'V21','V22','V23','V24','V25','V26','V27','V28','Amount']
#
# # ==========================
# # CSV Upload Section
# # ==========================
# if option == "Upload CSV":
#     uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
#     if uploaded_file:
#         data = pd.read_csv(uploaded_file)
#         st.subheader("Preview of Uploaded Data")
#         st.dataframe(data.head())
#
#         # Check for required features
#         if not all(f in data.columns for f in FEATURES):
#             st.error(f"Your CSV must contain all features: {FEATURES}")
#         else:
#             if st.button("Predict Fraud 🚨"):
#                 predictions = model.predict(data[FEATURES])
#                 probabilities = model.predict_proba(data[FEATURES])[:, 1]
#                 data["Prediction"] = predictions
#                 data["Fraud Probability"] = probabilities
#
#                 # Color-coded display
#                 def color_pred(val):
#                     return 'background-color: #ff9999' if val == 1 else 'background-color: #99ff99'
#
#                 st.subheader("Prediction Results")
#                 st.dataframe(data.style.applymap(color_pred, subset=["Prediction"]))
#
#                 # Summary metrics
#                 st.subheader("Summary Metrics")
#                 total = len(data)
#                 fraud_count = data["Prediction"].sum()
#                 non_fraud = total - fraud_count
#
#                 col1, col2, col3 = st.columns(3)
#                 col1.metric("Total Transactions", total)
#                 col2.metric("Fraudulent", fraud_count, delta=f"{(fraud_count / total) * 100:.2f}%")
#                 col3.metric("Non-Fraudulent", non_fraud, delta=f"{(non_fraud / total) * 100:.2f}%")
#
#                 # Probability distribution
#                 st.subheader("Fraud Probability Distribution")
#                 fig, ax = plt.subplots()
#                 sns.histplot(probabilities, bins=20, kde=True, color='skyblue', ax=ax)
#                 ax.set_xlabel("Fraud Probability")
#                 ax.set_ylabel("Count")
#                 st.pyplot(fig)
#     else:
#         st.info("Please upload a CSV file to make predictions.")
#
# # ==========================
# # Manual Input Section
# # ==========================
# elif option == "Manual Input":
#     st.subheader("Enter Transaction Details Manually")
#
#     # Dynamically create input fields for all 30 features
#     user_input = {}
#     for feature in FEATURES:
#         if feature == "Time":
#             user_input[feature] = st.number_input(feature, min_value=0, value=0)
#         elif feature == "Amount":
#             user_input[feature] = st.number_input(feature, min_value=0.0, value=0.0, format="%.2f")
#         else:
#             user_input[feature] = st.number_input(feature, value=0.0, format="%.6f")
#
#     if st.button("Predict Single Transaction 🚨"):
#         input_data = pd.DataFrame([user_input])
#         prediction = model.predict(input_data)[0]
#         prob = model.predict_proba(input_data)[0, 1]
#
#         st.subheader("Prediction Result")
#         if prediction == 1:
#             st.error(f"🚨 Fraudulent Transaction! Probability: {prob:.2f}")
#         else:
#             st.success(f"✅ Non-Fraudulent Transaction. Probability: {prob:.2f}")
# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    model_path = r"C:\Users\HP\Desktop\movi\fraud_model.pkl"
    model = joblib.load(model_path)
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
# Expected Features
# ==========================
FEATURES = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9',
            'V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
            'V21','V22','V23','V24','V25','V26','V27','V28','Amount']

# ==========================
# CSV Upload Section
# ==========================
if option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(data.head())

        # Check for required features
        if not all(f in data.columns for f in FEATURES):
            st.error(f"Your CSV must contain all features: {FEATURES}")
        else:
            if st.button("Predict Fraud 🚨"):
                predictions = model.predict(data[FEATURES])
                probabilities = model.predict_proba(data[FEATURES])[:, 1]
                data["Prediction"] = predictions
                data["Fraud Probability"] = probabilities

                # Color-coded display (preview only first 100 rows)
                def color_pred(val):
                    return 'background-color: #ff9990' if val == 1 else 'background-color: black'

                st.subheader("Prediction Results (Preview 100 rows)")
                st.dataframe(data.head(100).style.applymap(color_pred, subset=["Prediction"]))

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

    # Dynamically create input fields for all 30 features
    user_input = {}
    for feature in FEATURES:
        if feature == "Time":
            user_input[feature] = st.number_input(feature, min_value=0, value=0)
        elif feature == "Amount":
            user_input[feature] = st.number_input(feature, min_value=0.0, value=0.0, format="%.2f")
        else:
            user_input[feature] = st.number_input(feature, value=0.0, format="%.6f")

    if st.button("Predict Single Transaction 🚨"):
        input_data = pd.DataFrame([user_input])
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0, 1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"🚨 Fraudulent Transaction! Probability: {prob:.2f}")
        else:
            st.success(f"✅ Non-Fraudulent Transaction. Probability: {prob:.2f}")
