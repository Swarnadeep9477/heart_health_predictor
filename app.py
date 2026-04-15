import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Input fields
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

st.title("❤️ Heart Disease Prediction System")

st.markdown("### 🧾 Patient Information")

col1, col2 = st.columns(2)
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This app predicts heart disease using a Machine Learning model."
)

st.sidebar.markdown("### 🔧 Model Info")
st.sidebar.write("Algorithm: Logistic Regression")

with col1:
    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if "Male" else 0
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.number_input("Resting BP")
    fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", [0, 1])
    restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])

with col2:
    chol=st.number_input("Cholesterol", help="mg/dL")
    thalach = st.number_input("Max Heart Rate")
    oldpeak = st.number_input("Oldpeak")
    ca = st.selectbox("Major Vessels", [0,1,2,3])
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    slope = st.selectbox("Slope (0-2)", [0, 1, 2])
    thal = st.selectbox("Thal (0-2)", [0, 1, 2])

# Prediction button
if st.button("Predict"):
    input_data = np.array([
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]).reshape(1, -1)

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    st.markdown("---")

    if prediction[0] == 0:
        st.success(f"✅ No Heart Disease\n\nConfidence: {round(prob[0][0]*100,2)}%")
    else:
        st.error(f"⚠️ Heart Disease Detected\n\nConfidence: {round(prob[0][1]*100,2)}%")
    import pandas as pd

    prob_df = pd.DataFrame({
        "Condition": ["No Disease", "Disease"],
        "Probability": prob[0]
    })

    st.bar_chart(prob_df.set_index("Condition"))

    # Small-range (categorical encoded)
    small_features = {
        "Sex": sex,
        "Chest Pain": cp,
        "FBS": fbs,
        "Rest ECG": restecg,
        "Exercise Angina": exang,
        "Slope": slope,
        "CA": ca,
        "Thal": thal
    }

    # Large numeric values
    large_features = {
        "Age": age,
        "BP": trestbps,
        "Cholesterol": chol,
        "Max HR": thalach,
        "Oldpeak": oldpeak
    }

    st.markdown("### 📊 Key Health Metrics")

    cols = st.columns(len(large_features))

    for i, (key, value) in enumerate(large_features.items()):
        cols[i].metric(label=key, value=value)


    st.markdown("### 📉 Encoded Health Indicators")

    df = pd.DataFrame({
        "Feature": list(small_features.keys()),
        "Value": list(small_features.values())
    })

    st.bar_chart(df.set_index("Feature"))