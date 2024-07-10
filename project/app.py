# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:22:31 2024

@author: Impana
"""

import streamlit as st
from streamlit_option_menu import option_menu
from heart import app as heart_app
from kidney_stone_detection import app as kidney_app
from pneumonia import app as pneumonia_app
from diabetes import app as diabetes_app
from liver import app as liver_app
from home import app as home_app
from brain_tumor import app as brain_tumor_app

# Title and description
#st.title('Multiple Disease Prediction App')
#st.write('This app allows you to predict heart disease, diabetes, brain tumors, pneumonia, and more.')

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Heart Disease Prediction", "Diabetes Prediction", "Liver Disease Prediction", "Kidney Stone Detection", "Brain Tumor Detection", "Pneumonia Detection"],
        icons=["house", "heart", "activity", "thermometer", "droplet", "brain", "lungs"],
        menu_icon="cast",
        default_index=0,
    )

# Display selected page based on current_page
if selected == "Home":
    home_app()
elif selected == "Heart Disease Prediction":
    heart_app()
elif selected == "Diabetes Prediction":
    diabetes_app()
elif selected == "Liver Disease Prediction":
    liver_app()
elif selected == "Kidney Stone Detection":
    kidney_app()
elif selected == "Brain Tumor Detection":
    brain_tumor_app()
elif selected == "Pneumonia Detection":
    pneumonia_app()
