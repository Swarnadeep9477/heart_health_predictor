import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="heart",
    layout="centered",
)

st.title("Heart Disease Predictor")
st.write("Enter patient details and run the prediction.")


MODEL_META = {
    "Logistic Regression": {"file": "logistic_regression_model.pkl", "tier": "baseline"},
    "Random Forest": {"file": "random_forest_model.pkl", "tier": "high"},
    "KNN": {"file": "knn_model.pkl", "tier": "baseline"},
    "Naive Bayes": {"file": "naive_bayes_model.pkl", "tier": "baseline"},
    "Decision Tree": {"file": "decision_tree_model.pkl", "tier": "high"},
    "Perceptron": {"file": "perceptron_model.pkl", "tier": "baseline"},
    "XGBoost": {"file": "xgboost_model.pkl", "tier": "high"},
}

HIGH_ACCURACY_MODELS = ["XGBoost", "Random Forest", "Decision Tree"]


@st.cache_resource
def load_models():
    loaded = {}
    for name, meta in MODEL_META.items():
        try:
            loaded[name] = joblib.load(meta["file"])
        except Exception:
            pass
    return loaded


def preprocess_input(df):
    df = pd.get_dummies(df, drop_first=True)
    required_cols = [
        "age", "trestbps", "chol", "restecg", "thalach", "oldpeak", "ca",
        "sex_1", "cp_1", "cp_2", "cp_3", "fbs_1", "exang_1",
        "slope_1", "slope_2", "thal_1", "thal_2", "thal_3",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    return df[required_cols]


models = load_models()

if not models:
    st.warning("No model files were found. Place the .pkl model files in the same folder as this app.")

with st.form("prediction_form"):
    st.subheader("Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-Anginal",
                3: "Asymptomatic",
            }[x],
        )
        trestbps = st.number_input("Resting BP", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Yes" if x else "No")

    with col2:
        restecg = st.selectbox(
            "Rest ECG",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Abnormality",
                2: "LV Hypertrophy",
            }[x],
        )
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "Yes" if x else "No")
        oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox(
            "ST Slope",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Upsloping",
                1: "Flat",
                2: "Downsloping",
            }[x],
        )
        ca = st.selectbox("Fluoroscopy Vessels", [0, 1, 2, 3, 4])
        thal = st.selectbox(
            "Thalassemia",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Unknown",
                1: "Normal",
                2: "Fixed Defect",
                3: "Reversible Defect",
            }[x],
        )

    predict_clicked = st.form_submit_button("Run Prediction")


if predict_clicked and models:
    sex_encoded = 1 if sex == "Male" else 0

    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex_encoded,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }])

    processed = preprocess_input(input_df)
    results = []

    for name, model in models.items():
        try:
            pred = model.predict(processed)[0]
            prob = model.predict_proba(processed)[0][1] if hasattr(model, "predict_proba") else 0.5
            results.append({
                "Model": name,
                "Prediction": "High Risk" if pred == 1 else "Low Risk",
                "Risk %": round(prob * 100, 1),
                "Confidence %": round((prob if pred == 1 else 1 - prob) * 100, 1),
            })
        except Exception:
            continue

    if not results:
        st.error("The models could not make a prediction for this input.")
    else:
        df_results = pd.DataFrame(results)

        st.subheader("Model Results")
        st.dataframe(df_results, use_container_width=True)

        st.subheader("Risk Bar Graph")
        chart_df = df_results.set_index("Model")[["Risk %"]]
        st.bar_chart(chart_df)

        st.subheader("High Accuracy Models")
        df_high_accuracy = df_results[df_results["Model"].isin(HIGH_ACCURACY_MODELS)]
        st.dataframe(df_high_accuracy, use_container_width=True)

st.caption("This app is for decision support only and does not replace professional medical diagnosis.")