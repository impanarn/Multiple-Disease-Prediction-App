# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:14:42 2024

@author: Impana
"""

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

def preprocess_image_file(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = img.reshape(-1, 150, 150, 1)
    return img

# Load the saved model
#@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('C:/Users/Impana/OneDrive/Desktop/vital_forecast/pneumonia_detection_model.h5')
    return model

model = load_model()

# Streamlit UI
st.title('Pneumonia Detection')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    image = preprocess_image_file(uploaded_file)
    
    # Make prediction
    prediction = model.predict(image)
    if prediction > 0.5:
        st.write('Prediction: PNEUMONIA')
    else:
        st.write('Prediction: NORMAL')
    
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)



