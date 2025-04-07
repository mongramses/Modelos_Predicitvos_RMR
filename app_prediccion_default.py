
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Predicción de Default", layout="centered")

st.title("🔍 Predicción de Incumplimiento de Pago (Default)")

st.markdown("""Este sistema permite ingresar la información de un cliente y predecir la probabilidad de que incurra en **default** usando un modelo de regresión logística entrenado previamente.""")

@st.cache_resource
def load_model():
    model = joblib.load("logistic_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_names.pkl")
    return model, scaler, features

model, scaler, feature_names = load_model()

with st.form("formulario_prediccion"):
    st.subheader("📋 Datos del cliente")
    age = st.slider("Edad", 18, 75, 35)
    income = st.number_input("Ingreso anual", 10000, 200000, 50000)
    loanamount = st.number_input("Monto del préstamo", 1000, 200000, 10000)
    creditscore = st.slider("Puntaje de crédito", 300, 850, 600)
    monthsemployed = st.slider("Meses empleados", 0, 480, 60)
    numcreditlines = st.slider("Líneas de crédito activas", 0, 20, 5)
    interestrate = st.slider("Tasa de interés (%)", 0.0, 35.0, 15.0)
    loanterm = st.selectbox("Plazo del préstamo (meses)", [12, 24, 36, 48, 60])
    dtiratio = st.slider("Relación deuda/ingreso (0-1)", 0.0, 1.0, 0.3)

    hasmortgage = st.checkbox("¿Tiene hipoteca?")
    hasdependents = st.checkbox("¿Tiene dependientes?")
    hascosigner = st.checkbox("¿Tiene codeudor?")

    education = st.selectbox("Nivel educativo", ["High School", "Bachelor's", "Master's"])
    employmenttype = st.selectbox("Tipo de empleo", ["Full-time", "Part-time", "Unemployed"])
    maritalstatus = st.selectbox("Estado civil", ["Single", "Married", "Divorced"])
    loanpurpose = st.selectbox("Propósito del préstamo", ["Other", "Auto", "Business", "Debt Consolidation"])

    submitted = st.form_submit_button("🔮 Predecir")

    if submitted:
        input_dict = {
            'age': age,
            'income': income,
            'loanamount': loanamount,
            'creditscore': creditscore,
            'monthsemployed': monthsemployed,
            'numcreditlines': numcreditlines,
            'interestrate': interestrate,
            'loanterm': loanterm,
            'dtiratio': dtiratio,
            'hasmortgage': int(hasmortgage),
            'hasdependents': int(hasdependents),
            'hascosigner': int(hascosigner),
            "education_Bachelor's": 0,
            "education_Master's": 0,
            'employmenttype_Part-time': 0,
            'employmenttype_Unemployed': 0,
            'maritalstatus_Married': 0,
            'maritalstatus_Single': 0,
            'loanpurpose_Auto': 0,
            'loanpurpose_Business': 0,
            'loanpurpose_Debt Consolidation': 0
        }

        if education == "Bachelor's":
            input_dict["education_Bachelor's"] = 1
        elif education == "Master's":
            input_dict["education_Master's"] = 1

        if employmenttype == "Part-time":
            input_dict["employmenttype_Part-time"] = 1
        elif employmenttype == "Unemployed":
            input_dict["employmenttype_Unemployed"] = 1

        if maritalstatus == "Married":
            input_dict["maritalstatus_Married"] = 1
        elif maritalstatus == "Single":
            input_dict["maritalstatus_Single"] = 1

        if loanpurpose == "Auto":
            input_dict["loanpurpose_Auto"] = 1
        elif loanpurpose == "Business":
            input_dict["loanpurpose_Business"] = 1
        elif loanpurpose == "Debt Consolidation":
            input_dict["loanpurpose_Debt Consolidation"] = 1

        input_df = pd.DataFrame([input_dict])[feature_names]
        input_scaled = scaler.transform(input_df)

        prob = model.predict_proba(input_scaled)[0][1]
        st.success(f"📈 Probabilidad estimada de default: {prob:.2%}")
