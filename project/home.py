# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 23:36:37 2024

@author: Impana
"""

import streamlit as st

def app():
    st.title('Welcome to Multiple Disease Prediction System')
    st.write("""
    This application helps predict various diseases using machine learning models. 
    Select a disease from the sidebar to get started.
    """)

if __name__ == '__main__':
    app()
