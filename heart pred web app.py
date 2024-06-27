# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 18:17:34 2024

@author: Impana
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model and scaler
loaded_model, loaded_scaler = pickle.load(open('C:/Users/Impana/Downloads/VitalForecast/heart_trained_model.sav', 'rb'))

# creating a function for Prediction
def heart_disease_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # scaling the input data
    input_data_scaled = loaded_scaler.transform(input_data_reshaped)

    # prediction
    prediction = loaded_model.predict(input_data_scaled)
    
    if (prediction[0] == 0):
        return 'The Person does not have Heart Disease'
    else:
        return 'The Person has Heart Disease'

def main():
    # giving a title
    st.title('Heart Disease Prediction')
    
    # getting the input data from the user
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
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        diagnosis = heart_disease_prediction([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
