# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 18:29:33 2024

@author: Impana
"""

import streamlit as st
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import io

def process_image(uploaded_image):
    img = cv.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv.IMREAD_GRAYSCALE)
    
    X = img.shape[0]
    copy = np.copy(img)
    
    # Step-by-step images and titles
    images = []
    titles = []
    
    # Original Image
    images.append(copy)
    titles.append("Original Image")
    
    # Original intensity histogram
    hist, bins = np.histogram(copy.flatten(), 256, [0, 256])
    
    # First enhancement
    blur = cv.GaussianBlur(copy, (5, 5), 2)
    enh = cv.add(copy, cv.add(blur, -100))
    #images.append(enh)
    #titles.append("First Enhancement")
    
    # Denoising
    median = cv.medianBlur(enh, 5)
    #images.append(median)
    #titles.append("Denoised")
    
    # Morphological Gradient
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    gradient = cv.morphologyEx(median, cv.MORPH_GRADIENT, kernel)
    #images.append(gradient)
    #titles.append("Morphological Gradient")
    
    # Second enhancement
    enh2 = cv.add(median, gradient)
    #images.append(enh2)
    #titles.append("Second Enhancement")
    
    # First thresholding
    t = np.percentile(enh2, 85)
    ret, th = cv.threshold(enh2, t, 255, cv.THRESH_BINARY)
    #images.append(th)
    #titles.append("First Thresholding")
    
    # Morphology operations
    kernel_c = cv.getStructuringElement(cv.MORPH_ELLIPSE, (int((5*X)/100), int((5*X)/100)))
    kernel_e = cv.getStructuringElement(cv.MORPH_ELLIPSE, (int((3*X)/100), int((3*X)/100)))
    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (int((7*X)/100), int((7*X)/100)))

    opening = cv.morphologyEx(th, cv.MORPH_OPEN, kernel_e)
    #images.append(opening)
    #titles.append("First Opening")
    
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel_c)
    #images.append(closing)
    #titles.append("Closing")
    
    erosion = cv.erode(closing, kernel_e, iterations=1)
    #images.append(erosion)
    #titles.append("First Erosion")
    
    dilation = cv.dilate(erosion, kernel_e, iterations=1)
    #images.append(dilation)
    #titles.append("Dilation")
    
    # Masking
    masked = cv.bitwise_and(copy, copy, mask=dilation)
    #images.append(masked)
    #titles.append("Masked")
    
    # Second round of morphology operations
    s_erosion = cv.erode(masked, kernel, iterations=1)
    #images.append(s_erosion)
    #titles.append("Second Erosion")
    
    final = cv.morphologyEx(s_erosion, cv.MORPH_OPEN, ker)
    #images.append(final)
    #titles.append("Second Opening")
    
    # Third enhancement
    blur3 = cv.GaussianBlur(final, (3, 3), 0)
    enh3 = cv.add(final, cv.add(blur3, -100))
    #images.append(enh3)
    #titles.append("Third Enhancement")
    
    # Second thresholding
    upper = np.percentile(enh3, 92)
    res = cv.inRange(enh3, 0, upper)
    #images.append(res)
    #titles.append("Second Thresholding")
    
    # Final morphology step
    fin = cv.morphologyEx(res, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (int((7*X)/100), int((7*X)/100))))
    #images.append(fin)
    #titles.append("Last Closing")
    
    # Contouring
    copy_rgb = cv.cvtColor(copy, cv.COLOR_GRAY2RGB)
    contours, hierarchy = cv.findContours(fin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 1:
        cnt = contours[1]
        if len(contours) > 2:
            cv.drawContours(copy_rgb, contours, 2, (0, 255, 0), 3)
        else:
            cv.drawContours(copy_rgb, contours, 1, (0, 255, 0), 3)
        
        area = int(cv.contourArea(cnt))
        perimeter = int(cv.arcLength(cnt, True))
        result_text = f"Area: {area} px\nPerimeter: {perimeter} px"
    else:
        result_text = "No tumor detected"
    
    images.append(copy_rgb)
    titles.append("Detected Tumor")

    return images, titles, result_text

def app():
    st.title("Kidney Stone Detection")
    

    uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

    if uploaded_file is not None:
        images, titles, result_text = process_image(uploaded_file)
    
        cols = st.columns(3)  # Change the number of columns based on how many images you want per row
        for i, (img, title) in enumerate(zip(images, titles)):
            col = cols[i % 3]
            col.image(img, caption=title, use_column_width=True)
    
        st.write(result_text)
