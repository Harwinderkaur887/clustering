import streamlit as st
import joblib
import numpy as np

# ---------------------------
# Safe Import for scikit-fuzzy
# ---------------------------
try:
    import skfuzzy as fuzz
except ImportError:
    fuzz = None

# ---------------------------
# Load Models & Scaler
# ---------------------------
@st.cache_resource
def load_models():
    try:
        kmeans = joblib.load("km.pkl")
    except:
        kmeans = None

    try:
        dbscan = joblib.load("cl.pkl")
    except:
        dbscan = None

    try:
        meanshift = joblib.load("mn.pkl")
    except:
        meanshift = None

    try:
        scaler = joblib.load("s.pkl")
    except:
        scaler = None

    try:
        fuzzy_cntr, _ = joblib.load("fuzzy.pkl")
    except:
        fuzzy_cntr = None

    return kmeans, dbscan, meanshift, scaler, fuzzy_cntr


kmeans, dbscan, meanshift, scaler, fuzzy_cntr = load_models()

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Clustering Prediction App", layout="centered")
st.title("🔍 Clustering Prediction App")
st.write("Enter feature values and select the clustering algorithm to get predictions.")

# ---------------------------
# Algorithm Options
# ---------------------------
algorithms = ["KMeans", "DBSCAN", "Mean Shift"]
if fuzz and fuzzy_cntr is not None:
    algorithms.append("Fuzzy C-Means")

algorithm = st.selectbox("Choose a clustering algorithm:", algorithms)

# ---------------------------
# Feature Input (update names)
# ---------------------------
feature_names = ["Feature 1", "Feature 2"]  # Change to real names
user_input = []

for feature in feature_names:
    val = st.number_input(f"Enter {feature}:", value=0.0)
    user_input.append(val)

# ---------------------------
# Predict Button
# ---------------------------
if st.button("Predict Cluster"):
    if scaler is None:
        st.error("❌ Scaler file not found. Cannot scale input.")
    else:
        X = np.array(user_input).reshape(1, -1)
        X_scaled = scaler.transform(X)

        cluster = None

        if algorithm == "KMeans":
            if kmeans:
                cluster = kmeans.predict(X_scaled)[0]
            else:
                st.error("❌ KMeans model not found.")

        elif algorithm == "DBSCAN":
            if dbscan:
                # DBSCAN has no 'predict', so fit_predict on single point
                cluster = dbscan.fit_predict(X_scaled)[0]
                if cluster == -1:
                    st.warning("⚠️ This point is considered noise by DBSCAN.")
            else:
                st.error("❌ DBSCAN model not found.")

        elif algorithm == "Mean Shift":
            if meanshift:
                try:
                    cluster = meanshift.predict(X_scaled)[0]
                except AttributeError:
                    cluster = meanshift.fit_predict(X_scaled)[0]
            else:
                st.error("❌ Mean Shift model not found.")

        elif algorithm == "Fuzzy C-Means":
            if fuzz and fuzzy_cntr is not None:
                u, _, _, _, _, _, _ = fuzz.cmeans_predict(
                    X_scaled.T, fuzzy_cntr, m=2, error=0.005, maxiter=1000
                )
                cluster = np.argmax(u, axis=0)[0]
            else:
                st.error("❌ Fuzzy C-Means is not available.")

        if cluster is not None:
            st.success(f"✅ Predicted Cluster: {cluster}")
