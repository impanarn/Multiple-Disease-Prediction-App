# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:20:49 2024

@author: Impana
"""

import pickle
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Load the model from disk
model_path = 'C:/Users/Impana/OneDrive/Desktop/vital_forecast/projects/models/brain tumor_model.sav'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

def predict_image(image):
    img_array = np.array(image.convert('L'))
    img_resized = cv2.resize(img_array, (200, 200))
    img_normalized = img_resized.reshape(1, -1) / 255.0
    prediction = model.predict(img_normalized)
    return prediction[0]

def app():
    st.title("Brain Tumor Detection")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        prediction = predict_image(image)
        st.write(f"Prediction: {'Tumor' if prediction == 1 else 'No Tumor'}")
