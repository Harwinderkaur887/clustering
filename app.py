import streamlit as st
import pandas as pd
import numpy as np

st.cache_resource
def load_model():
    kmeans = joblib.load("kmeans.pkl")
    scaler = joblib.load("scaler.pkl")
    return kmeans, scaler

st.set_page_config(page_title="CSV Cleaner App", layout="wide")

st.title("🧼 CSV Cleaning App using Streamlit")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Original Data Preview")
    st.write(df.head())

    st.subheader("🔍 Null Value Summary")
    st.write(df.isnull().sum())

    # Fill missing values
    if 'Ever_Married' in df.columns:
        df["Ever_Married"] = df["Ever_Married"].fillna(df["Ever_Married"].mode()[0])

    if 'Graduated' in df.columns:
        df["Graduated"] = df["Graduated"].fillna(df["Graduated"].mode()[0])

    if 'Profession' in df.columns:
        df["Profession"] = df["Profession"].fillna(df["Profession"].mode()[0])

    if 'Work_Experience' in df.columns:
        df["Work_Experience"] = df["Work_Experience"].fillna(int(df["Work_Experience"].mean()))

    if 'Family_Size' in df.columns:
        df["Family_Size"] = df["Family_Size"].fillna(df["Family_Size"].mode()[0])

    if 'Var_1' in df.columns:
        df["Var_1"] = df["Var_1"].fillna(df["Var_1"].mode()[0])

    st.success("✅ Null values filled successfully!")

    st.subheader("🧼 Cleaned Data Preview")
    st.write(df.head())

    st.subheader("⬇️ Download Cleaned CSV")
    st.download_button("Download", data=df.to_csv(index=False), file_name="cleaned_data.csv", mime="text/csv")
else:
    st.info("Please upload a CSV file to get started.")
