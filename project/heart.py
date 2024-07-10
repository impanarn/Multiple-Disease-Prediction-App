# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:19:49 2024

@author: Impana
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model and scaler
loaded_model, loaded_scaler = pickle.load(open('C:/Users/Impana/OneDrive/Desktop/vital_forecast/projects/models/heart_trained_model.sav', 'rb'))

# creating a function for Prediction
def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    input_data_scaled = loaded_scaler.transform(input_data_reshaped)
    prediction = loaded_model.predict(input_data_scaled)
    return 'The Person has Heart Disease' if prediction[0] == 1 else 'The Person does not have Heart Disease'

def app():
    st.title('Heart Disease Prediction')
    age = st.text_input('Age')
    sex = st.text_input('Sex (1 = male; 0 = female)')
    cp = st.text_input('Chest Pain Type (0-3)')
    trestbps = st.text_input('Resting Blood Pressure')
    chol = st.text_input('Serum Cholesterol in mg/dl')
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)')
    restecg = st.text_input('Resting Electrocardiographic Results (0-2)')
    thalach = st.text_input('Maximum Heart Rate Achieved')
    exang = st.text_input('Exercise Induced Angina (1 = yes; 0 = no)')
    oldpeak = st.text_input('ST Depression Induced by Exercise')
    slope = st.text_input('Slope of the Peak Exercise ST Segment (0-2)')
    ca = st.text_input('Number of Major Vessels Colored by Fluoroscopy (0-4)')
    thal = st.text_input('Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)')
    
    diagnosis = ''
    if st.button('Heart Disease Test Result'):
        diagnosis = heart_disease_prediction([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        
    st.success(diagnosis)
