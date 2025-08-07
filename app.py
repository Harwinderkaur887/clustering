import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="User Data Input", layout="centered")
st.title("📝 User Information Form")

# Input form
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
ever_married = st.selectbox("Ever Married?", ["Yes", "No"])
age = st.number_input("Age", min_value=0, max_value=120, step=1)
graduated = st.selectbox("Graduated?", ["Yes", "No"])
profession = st.selectbox("Profession", ["Engineer", "Healthcare", "Executive", "Marketing", "Other"])
work_experience = st.number_input("Work Experience (Years)", min_value=0, max_value=50, step=1)
spending_score = st.selectbox("Spending Score", ["Low", "Average", "High"])
family_size = st.number_input("Family Size", min_value=0, max_value=20, step=1)
var_1 = st.selectbox("Var_1", ["Cat_6", "Cat_1", "Cat_2", "Cat_3", "Cat_4", "Cat_5", "Other"])

# Submit
if st.button("Submit"):
    # Generate a dummy ID (optional: could be from timestamp or incremental counter)
    new_id = pd.Timestamp.now().value % 10**6

    new_data = {
        "ID": new_id,
        "Gender": gender,
        "Ever_Married": ever_married,
        "Age": age,
        "Graduated": graduated,
        "Profession": profession,
        "Work_Experience": work_experience,
        "Spending_Score": spending_score,
        "Family_Size": family_size,
        "Var_1": var_1
    }

    df = pd.DataFrame([new_data])

    # Show result
    st.success("✅ Data Submitted Successfully!")
    st.dataframe(df)

    # Save to CSV (append mode)
    csv_file = "user_submissions.csv"
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)
