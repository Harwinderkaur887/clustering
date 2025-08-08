import streamlit as st
import joblib
import numpy as np
import skfuzzy as fuzz

# ---------------------------
# Load Models & Scaler
# ---------------------------
@st.cache_resource
def load_models():
    kmeans = joblib.load("km.pkl")
    dbscan = joblib.load("cl.pkl")
    meanshift = joblib.load("mn.pkl")
    scaler = joblib.load("s.pkl")
    fuzzy_cntr, _ = joblib.load("fuzzy.pkl")  # You saved tuple (cntr, cluster_labels)
    return kmeans, dbscan, meanshift, scaler, fuzzy_cntr

kmeans, dbscan, meanshift, scaler, fuzzy_cntr = load_models()

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Clustering Prediction App", layout="centered")
st.title("🔍 Clustering Prediction App")
st.write("Enter feature values and select the clustering algorithm to get predictions.")

# ---------------------------
# User Select Algorithm
# ---------------------------
algorithm = st.selectbox(
    "Choose a clustering algorithm:",
    ("KMeans", "DBSCAN", "Fuzzy C-Means", "Mean Shift")
)

# ---------------------------
# Feature Input (Change to real feature names)
# ---------------------------
feature_names = ["Feature 1", "Feature 2"]  # Update with actual feature names from your dataset
user_input = []

for feature in feature_names:
    val = st.number_input(f"Enter {feature}:", value=0.0)
    user_input.append(val)

# ---------------------------
# Predict Button
# ---------------------------
if st.button("Predict Cluster"):
    X = np.array(user_input).reshape(1, -1)
    X_scaled = scaler.transform(X)

    if algorithm == "KMeans":
        cluster = kmeans.predict(X_scaled)[0]

    elif algorithm == "DBSCAN":
        # DBSCAN has no predict method — run fit_predict for this single point
        cluster = dbscan.fit_predict(X_scaled)[0]
        if cluster == -1:
            st.warning("⚠️ This point is considered noise by DBSCAN.")

    elif algorithm == "Fuzzy C-Means":
        # Using saved cluster centers (fuzzy_cntr)
        u, _, _, _, _, _, _ = fuzz.cmeans_predict(
            X_scaled.T, fuzzy_cntr, m=2, error=0.005, maxiter=1000
        )
        cluster = np.argmax(u, axis=0)[0]

    elif algorithm == "Mean Shift":
        cluster = meanshift.predict(X_scaled)[0]

    st.success(f"✅ Predicted Cluster: {cluster}")
