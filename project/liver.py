# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 23:19:19 2024

@author: Impana
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved model and scaler
with open('C:/Users/Impana/OneDrive/Desktop/vital_forecast/projects/models/liver_disease_detection.sav', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('C:/Users/Impana/OneDrive/Desktop/vital_forecast/projects/models/lscaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# Function for prediction
def liver_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    input_data_scaled = loaded_scaler.transform(input_data_reshaped)
    prediction = loaded_model.predict(input_data_scaled)
    return 'The person has liver disease' if prediction[0] == 1 else 'The person does not have liver disease'

def app():
    st.title('Liver Disease Prediction')

    # Collecting user input
    age = st.text_input('Age')
    gender = st.text_input('Gender (1 = male, 0 = female)')
    total_bilirubin = st.text_input('Total Bilirubin')
    direct_bilirubin = st.text_input('Direct Bilirubin')
    alkaline_phosphotase = st.text_input('Alkaline Phosphotase')
    alamine_aminotransferase = st.text_input('Alamine Aminotransferase')
    aspartate_aminotransferase = st.text_input('Aspartate Aminotransferase')
    total_proteins = st.text_input('Total Proteins')
    albumin = st.text_input('Albumin')
    albumin_and_globulin_ratio = st.text_input('Albumin and Globulin Ratio')

    # Initialize diagnosis
    diagnosis = ''

    # Button for prediction
    if st.button('Liver Disease Test Result'):
        diagnosis = liver_disease_prediction([age, gender, total_bilirubin, direct_bilirubin,
                                              alkaline_phosphotase, alamine_aminotransferase,
                                              aspartate_aminotransferase, total_proteins, albumin,
                                              albumin_and_globulin_ratio])
        
    st.success(diagnosis)

if __name__ == '__main__':
    app()
