import os
import cv2
import torch
import numpy as np
from flask import Flask, request, render_template
from PIL import Image

app = Flask(__name__)

# Load YOLOv5 model
MODEL_PATH = 'trained_weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

# Folder to save detected images
DETECTED_FOLDER = 'static/detected'

# Ensure the folder exists
if not os.path.exists(DETECTED_FOLDER):
    os.makedirs(DETECTED_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Read the uploaded image
            img = Image.open(uploaded_file)
            img_array = np.array(img)

            # Run detection on the image
            results = model(img_array)

            # Convert results to a pandas dataframe for easier processing
            detections = results.pandas().xyxy[0]  # [xmin, ymin, xmax, ymax, confidence, class, name]

            if not detections.empty:
                # Find the detection with the highest confidence
                highest_confidence_detection = detections.iloc[detections['confidence'].idxmax()]

                # Extract bounding box coordinates
                xmin = int(highest_confidence_detection['xmin'])
                ymin = int(highest_confidence_detection['ymin'])
                xmax = int(highest_confidence_detection['xmax'])
                ymax = int(highest_confidence_detection['ymax'])

                # Crop the image to the bounding box of the license plate
                cropped_image = img_array[ymin:ymax, xmin:xmax]

                # Convert cropped image back to PIL Image
                cropped_pil_image = Image.fromarray(cropped_image)

                # Save the cropped image in the detected/ folder
                detected_image_path = os.path.join(DETECTED_FOLDER, 'detected_license_plate.jpg')
                cropped_pil_image.save(detected_image_path)

                # Render the template with the detected image
                return render_template('index.html', detected_image=detected_image_path)

            else:
                return render_template('index.html', message="No license plate detected.")
        else:
            return render_template('index.html', message="No file uploaded.")
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
