# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:22:31 2024

@author: Impana
"""

import streamlit as st
from heart import app as heart_app
from brain_tumor import app as brain_tumor_app
from pneumonia import app as pneumonia_app

# Title and description
st.title('Multiple Disease Prediction App')
st.write('This app allows you to predict heart disease, brain tumors, and pneumonia.')

# Sidebar navigation
st.sidebar.title("Navigation")

# Use session state for page management
current_page = st.session_state.get("current_page", "home")

# Sidebar buttons for navigation
if st.sidebar.button("Home"):
    current_page = "home"

if st.sidebar.button("Heart Disease Prediction"):
    current_page = "heart"

if st.sidebar.button("Brain Tumor Detection"):
    current_page = "brain_tumor"

if st.sidebar.button("Pneumonia Detection"):
    current_page = "pneumonia"

# Update session state
st.session_state["current_page"] = current_page

# Display selected page based on current_page
if current_page == "home":
    # Home page content
    pass
elif current_page == "heart":
    heart_app()
elif current_page == "brain_tumor":
    brain_tumor_app()
elif current_page == "pneumonia":
    pneumonia_app()


