# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:21:34 2024

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

def load_model():
    model = tf.keras.models.load_model('C:/Users/Impana/OneDrive/Desktop/vital_forecast/projects/models/pneumonia_detection_model.h5')
    return model

model = load_model()

def app():
    st.title('Pneumonia Detection')
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = preprocess_image_file(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)
        prediction = model.predict(image)
        st.write('Prediction: PNEUMONIA' if prediction > 0.5 else 'Prediction: NORMAL')
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
