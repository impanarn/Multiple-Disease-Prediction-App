# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:10:04 2024

@author: Impana
"""

import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('C:/Users/Impana/OneDrive/Desktop/vital_forecast/pneumonia_detection_model.h5')

# Function to preprocess an image for prediction
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150), color_mode='grayscale')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    return img_array / 255.0

# Example usage: Predict on a single image
image_path = 'C:/Users/Impana/OneDrive/Desktop/vital_forecast/chest_xray/test/PNEUMONIA/person1_virus_8.jpeg'
processed_img = preprocess_image(image_path)
prediction = model.predict(processed_img)
print(f"Prediction: {'PNEUMONIA' if prediction > 0.5 else 'NORMAL'}")
