# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:19:49 2024

@author: Impana
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('C:/Users/Impana/OneDrive/Desktop/vital_forecast/projects/models/kidney_model.sav', 'rb'))

# Function for Kidney Disease Prediction
def kidney_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return 'The Person has Kidney Disease' if prediction[0] == 1 else 'The Person does not have Kidney Disease'

def app():
    st.title('Kidney Disease Prediction')
    
    age = st.text_input('Age')
    blood_pressure = st.text_input('Blood Pressure')
    specific_gravity = st.text_input('Specific Gravity')
    albumin = st.text_input('Albumin')
    sugar = st.text_input('Sugar')
    red_blood_cells = st.text_input('Red Blood Cells (normal=0; abnormal=1)')
    pus_cell = st.text_input('Pus Cell (normal=0; abnormal=1)')
    pus_cell_clumps = st.text_input('Pus Cell Clumps (notpresent=0; present=1)')
    bacteria = st.text_input('Bacteria (notpresent=0; present=1)')
    blood_glucose_random = st.text_input('Blood Glucose Random')
    blood_urea = st.text_input('Blood Urea')
    serum_creatinine = st.text_input('Serum Creatinine')
    sodium = st.text_input('Sodium')
    potassium = st.text_input('Potassium')
    haemoglobin = st.text_input('Haemoglobin')
    packed_cell_volume = st.text_input('Packed Cell Volume')
    white_blood_cell_count = st.text_input('White Blood Cell Count')
    red_blood_cell_count = st.text_input('Red Blood Cell Count')
    hypertension = st.text_input('Hypertension (no=0; yes=1)')
    diabetes_mellitus = st.text_input('Diabetes Mellitus (no=0; yes=1)')
    coronary_artery_disease = st.text_input('Coronary Artery Disease (no=0; yes=1)')
    appetite = st.text_input('Appetite (good=0; poor=1)')
    peda_edema = st.text_input('Pedal Edema (no=0; yes=1)')
    aanemia = st.text_input('Anemia (no=0; yes=1)')
    
    diagnosis = ''
    if st.button('Kidney Disease Test Result'):
        diagnosis = kidney_disease_prediction([
            age, blood_pressure, specific_gravity, albumin, sugar, red_blood_cells, 
            pus_cell, pus_cell_clumps, bacteria, blood_glucose_random, blood_urea, 
            serum_creatinine, sodium, potassium, haemoglobin, packed_cell_volume, 
            white_blood_cell_count, red_blood_cell_count, hypertension, diabetes_mellitus, 
            coronary_artery_disease, appetite, peda_edema, aanemia
        ])
        
    st.success(diagnosis)

