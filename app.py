import streamlit as st
import pandas as pd
import joblib

# Cargar modelo y scaler
modelo = joblib.load("modelo_rf_optimizado.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Predicción de Churn")

# Inputs del usuario
tenure = st.slider("Tenure (meses de permanencia)", 0, 72, 12)
monthlycharges = st.number_input("Monthly Charges", value=70.0)
totalcharges = st.number_input("Total Charges", value=845.0)
contract = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
gender = st.selectbox("Género", ["Male", "Female"])
payment = st.selectbox("Método de pago", [
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)"
])

input_data = {
    # variables numéricas
    'seniorcitizen': [0],
    'tenure': [tenure],
    'monthlycharges': [monthlycharges],
    'totalcharges': [totalcharges],

    # dummies de género
    'gender_Male': [1 if gender == "Male" else 0],

    # dummies de estado civil / dependientes (quedan como estaban)
    'partner_Yes': [1],
    'dependents_Yes': [0],

    # dummies de servicios (quedan como estaban)
    'phoneservice_Yes': [1],
    'multiplelines_No phone service': [0],
    'multiplelines_Yes': [1],
    'internetservice_Fiber optic': [1],
    'internetservice_No': [0],
    'onlinesecurity_No internet service': [0],
    'onlinesecurity_Yes': [0],
    'onlinebackup_No internet service': [0],
    'onlinebackup_Yes': [1],
    'deviceprotection_No internet service': [0],
    'deviceprotection_Yes': [1],
    'techsupport_No internet service': [0],
    'techsupport_Yes': [0],
    'streamingtv_No internet service': [0],
    'streamingtv_Yes': [1],
    'streamingmovies_No internet service': [0],
    'streamingmovies_Yes': [1],

    # dummies de contrato
    'contract_One year': [1 if contract == "One year" else 0],
    'contract_Two year': [1 if contract == "Two year" else 0],

    # dummies de facturación
    'paperlessbilling_Yes': [1],

    # dummies de método de pago
    'paymentmethod_Credit card (automatic)': [1 if payment == "Credit card (automatic)" else 0],
    'paymentmethod_Electronic check':        [1 if payment == "Electronic check"        else 0],
    'paymentmethod_Mailed check':            [1 if payment == "Mailed check"            else 0],
    'paymentmethod_Bank transfer (automatic)': [1 if payment == "Bank transfer (automatic)" else 0],
}

df = pd.DataFrame(input_data)
df[['tenure', 'monthlycharges', 'totalcharges']] = scaler.transform(
    df[['tenure', 'monthlycharges', 'totalcharges']]
)

if st.button("Predecir Churn"):
    prob = modelo.predict_proba(df)[0, 1]
    st.success(f"Probabilidad de churn: **{prob:.2%}**")
