import streamlit as st
import pandas as pd
import joblib

# ================= CONFIG =================
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center;'>❤️ Heart Disease Prediction System</h1>
    <p style='text-align: center;'>Compare multiple ML models in one place</p>
    """,
    unsafe_allow_html=True
)

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("logistic_regression_model.pkl"),
        "Random Forest": joblib.load("random_forest_model.pkl"),
        "KNN": joblib.load("knn_model.pkl"),
        "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
        "Decision Tree": joblib.load("decision_tree_model.pkl"),
        "Perceptron": joblib.load("perceptron_model.pkl"),
        "XGBoost": joblib.load("xgboost_model.pkl"),
        "Voting Classifier": joblib.load("voting_classifier_model.pkl"),
    }

models = load_models()

# ================= INPUT =================
st.subheader("🧾 Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting BP", 80, 200, 120)

with col2:
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("FBS > 120", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.number_input("Max HR", 60, 220, 150)

with col3:
    exang = st.selectbox("Exercise Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("CA", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

sex = 1 if sex == "Male" else 0

# ================= PREPROCESS =================
def preprocess_input(df):
    df = pd.get_dummies(df, drop_first=True)

    required_cols = [
        'age','trestbps','chol','restecg','thalach','oldpeak','ca',
        'sex_1','cp_1','cp_2','cp_3',
        'fbs_1','exang_1',
        'slope_1','slope_2',
        'thal_1','thal_2','thal_3'
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    return df[required_cols]

# ================= PREDICTION =================
if st.button("🚀 Predict"):

    input_df = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg,
        "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope,
        "ca": ca, "thal": thal
    }])

    processed = preprocess_input(input_df)

    results = []

    for name, model in models.items():
        try:
            pred = model.predict(processed)[0]

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(processed)[0][1]
            else:
                prob = 0.5  # fallback

            results.append({
                "Model": name,
                "Prediction": "High Risk" if pred == 1 else "Low Risk",
                "Probability": prob
            })

        except:
            continue

    df_results = pd.DataFrame(results)

    # ================= DISPLAY =================
    st.subheader("📊 Model Comparison")

    # Table
    st.dataframe(df_results, use_container_width=True)

    # Highlight best model
    best_model = df_results.iloc[df_results["Probability"].idxmax()]

    st.subheader("🏆 Most Confident Model")
    st.info(f"{best_model['Model']} → {best_model['Prediction']} ({best_model['Probability']:.2f})")

    # ================= VISUAL BARS =================
    st.subheader("📈 Probability Comparison")

    for _, row in df_results.iterrows():
        st.write(f"**{row['Model']}**")
        st.progress(float(row["Probability"]))

    # ================= FINAL DECISION =================
    avg_prob = df_results["Probability"].mean()

    st.subheader("🧠 Final Ensemble Insight")

    if avg_prob > 0.5:
        st.error(f"⚠️ Overall High Risk (Avg Prob: {avg_prob:.2f})")
    else:
        st.success(f"✅ Overall Low Risk (Avg Prob: {1-avg_prob:.2f})")

    # ================= HIGH ACCURACY MODELS =================

    st.subheader("🔥 High Accuracy Models Insight")

    high_models = ["XGBoost", "Random Forest", "Voting Classifier"]

    df_high = df_results[df_results["Model"].isin(high_models)]

    if not df_high.empty:

        st.dataframe(df_high, use_container_width=True)

        # Best among high accuracy models
        best_high = df_high.loc[df_high["Probability"].idxmax()]

        st.info(
            f"🎯 Best High-Accuracy Model: {best_high['Model']} → "
            f"{best_high['Prediction']} ({best_high['Probability']:.2f})"
        )

        # Combined insight (weighted thinking)
        avg_high = df_high["Probability"].mean()

        if avg_high > 0.5:
            st.error(f"⚠️ High Risk (High-Model Avg: {avg_high:.2f})")
        else:
            st.success(f"✅ Low Risk (Confidence: {1 - avg_high:.2f})")

    st.subheader("📊 Probability Comparison (Bar Chart)")

    chart_df = df_results.copy()
    chart_df = chart_df.sort_values(by="Probability", ascending=False)
    chart_df = chart_df.set_index("Model")

    st.bar_chart(chart_df["Probability"])