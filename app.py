import streamlit as st
import joblib
import numpy as np

# ---------------------------
# Load Models & Scaler
# ---------------------------
@st.cache_resource
def load_models():
    kmeans = joblib.load("km.pkl")
    dbscan = joblib.load("cl.pkl")
    fcm = joblib.load("fuzzy.pkl")  # Assuming this is a fitted model object
    meanshift = joblib.load("mn.pkl")
    scaler = joblib.load("s.pkl")
    return kmeans, dbscan, fcm, meanshift, scaler

kmeans, dbscan, fcm, meanshift, scaler = load_models()

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
# Feature Input (Change to your real feature names)
# ---------------------------
feature_names = ["Feature 1", "Feature 2"]  # Replace with real feature names
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
        cluster = dbscan.fit_predict(X_scaled)[0]  # Or use precomputed if available
        if cluster == -1:
            st.warning("⚠️ This point is considered noise by DBSCAN.")
    
    elif algorithm == "Fuzzy C-Means":
        # Assuming fcm has a predict method or u, _, _, _, _ = fcm.predict(X_scaled)
        u, _, _, _, _, _, _ = fcm.predict(X_scaled)
        cluster = np.argmax(u, axis=1)[0]
    
    elif algorithm == "Mean Shift":
        cluster = meanshift.predict(X_scaled)[0]

    st.success(f"✅ Predicted Cluster: {cluster}")
