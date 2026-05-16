
import streamlit as st
import pandas as pd
import joblib
import os
import sklearn # Import scikit-learn to check its version

st.write(f"Scikit-learn version in Streamlit app: {sklearn.__version__}") # Display version

# Load the model and preprocessor
# Assuming these files are in the current working directory as set by the notebook
@st.cache_resource
def load_model_and_pipeline():
    try:
        modelo_churn = joblib.load('modelo_churn.pkl')
        pipeline_preproc = joblib.load('pipeline_preproc.pkl')
        return modelo_churn, pipeline_preproc
    except FileNotFoundError:
        st.error("Model or pipeline files not found. Make sure 'modelo_churn.pkl' and 'pipeline_preproc.pkl' are in the same directory as this app.")
        st.stop()
    except AttributeError as e:
        st.error(f"Error loading model or pipeline due to version incompatibility: {e}. This often happens when models are saved with one scikit-learn version and loaded with another. Please ensure your models were re-saved with compatible library versions and that the latest .pkl files are deployed.")
        st.stop()
    except Exception as e: # Catch any other unexpected errors during loading
        st.error(f"An unexpected error occurred while loading models: {e}")
        st.stop()

modelo_churn, pipeline_preproc = load_model_and_pipeline()

st.set_page_config(page_title="Eco-Ride Churn Prediction", layout="wide")
st.title("Eco-Ride Subscription Churn Prediction")
st.write("Enter customer details to predict if they will churn from the Eco-Ride subscription.")

# Input fields for features, updated based on the error message
st.sidebar.header("Customer Input Features")

edad = st.sidebar.slider('Edad', 18, 90, 30)
plan = st.sidebar.selectbox('Plan', ['Basic', 'Standard', 'Premium', 'Pro'])
region = st.sidebar.selectbox('Region', ['Urban', 'Suburban', 'Rural', 'North', 'South', 'East', 'West'])
uso_mensual_km = st.sidebar.number_input('Uso Mensual (Km)', 0.0, 5000.0, 500.0, step=10.0)
soporte_tickets = st.sidebar.slider('Tickets de Soporte', 0, 20, 2)
dias_antiguedad = st.sidebar.slider('Días de Antigüedad', 0, 365, 90)
gasto_promedio = st.sidebar.number_input('Gasto Promedio', 0.0, 1000.0, 100.0, step=10.0)

# Create a DataFrame from inputs with the correct column names
input_data = pd.DataFrame({
    'Edad': [edad],
    'Plan': [plan],
    'Region': [region],
    'Uso_Mensual_Km': [uso_mensual_km],
    'Soporte_Tickets': [soporte_tickets],
    'Dias_Antiguedad': [dias_antiguedad],
    'Gasto_Promedio': [gasto_promedio]
})

st.subheader("Input Data Preview")
st.dataframe(input_data)

if st.button('Predict Churn'):
    # Preprocess the input data
    try:
        processed_data = pipeline_preproc.transform(input_data)
        
        # Make prediction
        prediction = modelo_churn.predict(processed_data)
        prediction_proba = modelo_churn.predict_proba(processed_data)[:, 1] # Probability of churn

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"The customer is predicted to CHURN with a probability of {prediction_proba[0]:.2f}.")
        else:
            st.success(f"The customer is predicted NOT to churn with a probability of {1 - prediction_proba[0]:.2f}.")
        st.write(f"Churn Probability: {prediction_proba[0]:.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure the input features match the model's expectations after preprocessing.")
