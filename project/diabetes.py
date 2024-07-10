# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 22:17:56 2024

@author: Impana
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved model and scaler
with open('C:/Users/Impana/OneDrive/Desktop/vital_forecast/projects/models/diabetes_detection_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('C:/Users/Impana/OneDrive/Desktop/vital_forecast/projects/models/scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# Function for prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    input_data_scaled = loaded_scaler.transform(input_data_reshaped)
    prediction = loaded_model.predict(input_data_scaled)
    return 'The person has diabetes' if prediction[0] == 1 else 'The person does not have diabetes'

def app():
    st.title('Diabetes Prediction')

    # Collecting user input
    pregnancies = st.text_input('Number of Pregnancies')
    glucose = st.text_input('Glucose Level')
    blood_pressure = st.text_input('Blood Pressure Level')
    skin_thickness = st.text_input('Skin Thickness')
    insulin = st.text_input('Insulin Level')
    bmi = st.text_input('BMI')
    dpf = st.text_input('Diabetes Pedigree Function')
    age = st.text_input('Age')

    # Initialize diagnosis
    diagnosis = ''

    # Button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age])
        
    st.success(diagnosis)

if __name__ == '__main__':
    app()
