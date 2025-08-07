import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# Load the saved model and scaler
kmeans = joblib.load("kmeans_model.pkl")  # Make sure this file is present
scaler = joblib.load("scaler.pkl")        # Make sure this file is present

# Set Streamlit configuration
st.set_page_config(page_title="KMeans Clustering", layout="centered")
st.title("📊 User Cluster Prediction using KMeans")

# Input form
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
ever_married = st.selectbox("Ever Married?", ["Yes", "No"])
age = st.number_input("Age", min_value=0, max_value=120, step=1)
graduated = st.selectbox("Graduated?", ["Yes", "No"])
profession = st.selectbox("Profession", ["Engineer", "Healthcare", "Executive", "Marketing", "Other"])
work_experience = st.number_input("Work Experience (Years)", min_value=0, max_value=50, step=1)
spending_score = st.selectbox("Spending Score", ["Low", "Average", "High"])
family_size = st.number_input("Family Size", min_value=0, max_value=20, step=1)
var_1 = st.selectbox("Var_1", ["Cat_1", "Cat_2", "Cat_3", "Cat_4", "Cat_5", "Cat_6", "Other"])

if st.button("Predict Cluster"):
    # Create input DataFrame
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Ever_Married": ever_married,
        "Age": age,
        "Graduated": graduated,
        "Profession": profession,
        "Work_Experience": work_experience,
        "Spending_Score": spending_score,
        "Family_Size": family_size,
        "Var_1": var_1
    }])

    # Encode categorical variables using get_dummies (same as training)
    input_encoded = pd.get_dummies(input_data)

    # Align with model training columns
    expected_cols = scaler.feature_names_in_
    for col in expected_cols:
        if col not in input_encoded:
            input_encoded[col] = 0
    input_encoded = input_encoded[expected_cols]

    # Scale the data
    input_scaled = scaler.transform(input_encoded)

    # Predict cluster
    cluster = kmeans.predict(input_scaled)[0]

    # Output
    st.success(f"🎯 Predicted Cluster: {cluster}")
    input_data["Predicted_Cluster"] = cluster
    st.dataframe(input_data)

    # Save to CSV
    if os.path.exists("user_predictions.csv"):
        input_data.to_csv("user_predictions.csv", mode='a', header=False, index=False)
    else:
        input_data.to_csv("user_predictions.csv", mode='w', header=True, index=False)
