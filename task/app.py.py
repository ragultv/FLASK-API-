from ultralytics import YOLO
from flask import Flask, request, send_file
from PIL import Image
from retinaface import RetinaFace
import numpy as np
import io
import os
import tempfile
import cv2
from lwcc import LWCC
model = YOLO(r"C:\Users\tragu\Downloads\best.pt")
model1=YOLO('yolov8n.pt')
app = Flask('__name__')

def recognition(img):
    results = RetinaFace.detect_faces(img)
    return results
def predict(img):
    results = model.predict(img)
    return results  # Return the prediction results

def object_detection(img):
    results=model1(img)
    return results

def crowd_count(img):

    results=LWCC.get_count(img)

    return results

@app.route('/predict', methods=['POST'])
def predict_route():
    if request.method == 'POST':
        image = request.files.get('file')
        converted = Image.open(image)
        imgarr = np.array(converted)
        results = predict(imgarr)

        result_img = results[0].plot()
        result_img_pil = Image.fromarray(result_img)
        img_io = io.BytesIO()
        result_img_pil.save(img_io, 'JPEG')
        img_io.seek(0)

        # Return the image with predictions
        return send_file(img_io, mimetype='image/jpeg')
@app.route('/recog',methods=['POST'])
def recog():
    if request.method == 'POST':
        image=request.files.get('file')
        converted=Image.open(image)
        imgarr=np.array(converted)
        results= recognition(imgarr)

        result_img_pil = np.array(converted.copy())
        for face in results.values():
            facial_area = face['facial_area']
            cv2.rectangle(result_img_pil, (facial_area[0], facial_area[1]),
                          (facial_area[2], facial_area[3]), (255, 0, 0), 2)


        result_img_pil = Image.fromarray(result_img_pil)
        img_io = io.BytesIO()
        result_img_pil.save(img_io, 'JPEG')
        img_io.seek(0)

        # Return the image with recognition
        return send_file(img_io, mimetype='image/jpeg')

@app.route('/detect', methods=['POST'])
def detect_route():
    if request.method =='POST':
        image=request.files.get('file')
        converted=Image.open(image)
        imgarr=np.array(converted)
        results=object_detection(imgarr)
        # Assuming results[0].plot() returns a numpy array representing the image
        result_img = results[0].plot()
        result_img_pil = Image.fromarray(result_img)
        img_io = io.BytesIO()
        result_img_pil.save(img_io, 'JPEG')
        img_io.seek(0)
        # Return the image with predictions
        return send_file(img_io, mimetype='image/jpeg')

@app.route('/count',methods=['POST'])
def count():
    if request.method==('POST'):
        image=request.files.get('file')
        img1 = Image.open(image)
        img1_np = np.array(img1)  # Convert PIL Image to NumPy array
        # Save the grayscale image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_image_path = temp_file.name
            img1.save(temp_image_path)

        try:
            results = crowd_count(temp_image_path)

            img1_np_bgr = cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR)
            # Add text to the image (adjust position, font, etc. as needed)
            cv2.putText(img1_np_bgr, f"Count: {int(results)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert the OpenCV image back to a PIL image
            result_img_pil = Image.fromarray(cv2.cvtColor(img1_np_bgr, cv2.COLOR_BGR2RGB))
            img_io = io.BytesIO()
            result_img_pil.save(img_io, 'JPEG')
            img_io.seek(0)

            # Return the image with predictions
            return send_file(img_io, mimetype='image/jpeg')
        finally:
            # Clean up the temporary file
            os.remove(temp_image_path)


if __name__ == '__main__':
    app.run(debug=True)
