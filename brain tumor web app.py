# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 19:27:48 2024

@author: Impana
"""

import pickle
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Load the model from disk
model_path = 'C:/Users/Impana/OneDrive/Desktop/vital_forecast/brain tumor_model.sav'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Define a function to process and predict the uploaded image
def predict_image(image):
    # Convert the image to grayscale and resize it
    img_array = np.array(image.convert('L'))
    img_resized = cv2.resize(img_array, (200, 200))
    img_normalized = img_resized.reshape(1, -1) / 255.0  # Normalize

    # Predict using the loaded model
    prediction = model.predict(img_normalized)
    return prediction[0]

# Streamlit interface
st.title("Brain Tumor Detection")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict and display the result
    prediction = predict_image(image)
    st.write(f"Prediction: {'Tumor' if prediction == 1 else 'No Tumor'}")

