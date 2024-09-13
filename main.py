import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Load YOLOv5 model
MODEL_PATH = 'trained_weights/best.pt'
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit app UI
st.title('License Plate Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    try:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.stop()

    # Run detection on the uploaded image
    try:
        results = model(img_array)
        detections = results.pandas().xyxy[0]  # [xmin, ymin, xmax, ymax, confidence, class, name]
    except Exception as e:
        st.error(f"Error running detection: {e}")
        st.stop()

    if not detections.empty:
        # Find the detection with the highest confidence
        highest_confidence_detection = detections.iloc[detections['confidence'].idxmax()]

        # Extract bounding box of the detection
        xmin = int(highest_confidence_detection['xmin'])
        ymin = int(highest_confidence_detection['ymin'])
        xmax = int(highest_confidence_detection['xmax'])
        ymax = int(highest_confidence_detection['ymax'])

        # Crop the image to the bounding box of the license plate
        cropped_image = img_array[ymin:ymax, xmin:xmax]

        # Display the original image
        st.image(image, caption='Original Image', use_column_width=True)

        # Display the cropped license plate
        st.image(cropped_image, caption='Detected License Plate', use_column_width=True)
    else:
        st.write("No license plate detected.")
