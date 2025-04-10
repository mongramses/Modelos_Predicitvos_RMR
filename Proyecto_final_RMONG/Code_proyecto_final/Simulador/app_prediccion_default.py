
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

@st.cache_resource
def load_model():
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(ruta_actual, "logistic_model.pkl"))
    scaler = joblib.load(os.path.join(ruta_actual, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(ruta_actual, "feature_names.pkl"))
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

st.title("🔍 Predicción de Incumplimiento de Crédito")

# Entradas del usuario (solo las necesarias)
st.subheader("✍️ Información del solicitante")
Age = st.slider("Edad", 18, 100, 35)
Income = st.number_input("Ingreso mensual", min_value=0.0, max_value=20000.0, value=3500.0)
LoanAmount = st.number_input("Monto del préstamo", min_value=0.0, max_value=100000.0, value=15000.0)
CreditScore = st.slider("Puntaje crediticio", 300, 850, 680)
LoanTerm = st.selectbox("Plazo del préstamo (meses)", [12, 24, 36, 48, 60])
DTIRatio = st.slider("Relación deuda/ingreso (DTI)", min_value=0.0, max_value=5.0, value=1.0)
MonthsEmployed = st.slider("Meses en el empleo actual", 0, 480, 24)
NumCreditLines = st.number_input("Número de líneas de crédito activas", 0, 20, 3)
InterestRate = st.slider("Tasa de interés del préstamo (%)", 0.0, 100.0, 15.0)

# Diccionario corregido con nombres consistentes con el modelo
input_dict = {
    'Age': Age,
    'Income': Income,
    'LoanAmount': LoanAmount,
    'CreditScore': CreditScore,
    'MonthsEmployed': MonthsEmployed,
    'NumCreditLines': NumCreditLines,
    'InterestRate': InterestRate,
    'LoanTerm': LoanTerm,
    'DTIRatio': DTIRatio
}

input_df = pd.DataFrame([input_dict])

# Validación de columnas
missing_cols = [col for col in feature_names if col not in input_df.columns]
if missing_cols:
    st.error(f"❌ Faltan columnas requeridas por el modelo: {missing_cols}")
else:
    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)
    prob_default = model.predict_proba(input_scaled)[0][1]

    # Mostrar resultado
    st.subheader("📊 Resultado de la Predicción")
    st.metric("Probabilidad de Incumplimiento", f"{prob_default:.2%}")

    # Semáforo de riesgo
    if prob_default >= 0.8:
        st.error("🔴 Riesgo MUY ALTO de incumplimiento")
    elif prob_default >= 0.5:
        st.warning("🟠 Riesgo MEDIO de incumplimiento")
    else:
        st.success("🟢 Riesgo BAJO de incumplimiento")

    # Descarga CSV
    input_df['prob_default'] = prob_default
    csv = input_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar resultado en CSV",
        data=csv,
        file_name="prediccion_default.csv",
        mime='text/csv'
    )
