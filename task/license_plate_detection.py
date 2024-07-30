from ultralytics import YOLO
from flask import Flask, send_file, request
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import cv2
import pytesseract

model = YOLO(r"C:\Users\tragu\OneDrive\Documents\models\best.pt")

app = Flask('__name__')

def predict(img):
    results = model.predict(img)
    return results

def extract_license_plate_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 8')
    return text.strip()

@app.route('/predict', methods=['POST'])
def predict_route():
    if request.method == 'POST':
        image = request.files.get('file')
        converted = Image.open(image)
        imgarr = np.array(converted)
        results = predict(imgarr)

        detection = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        license_plate_img = imgarr[y1:y2, x1:x2]
        license_plate_text = extract_license_plate_text(license_plate_img)
        draw = ImageDraw.Draw(converted)
        font_path = r"C:\Users\tragu\Downloads\Humor-Sans.ttf"
        font = ImageFont.truetype(font_path, 24)
        text_bbox = font.getbbox(license_plate_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x2 - text_width
        text_y = y1 - text_height if y1 - text_height > 0 else y1

        draw.rectangle(((text_x, text_y), (text_x + text_width, text_y + text_height)), fill="black")
        draw.text((text_x, text_y), license_plate_text, fill="white", font=font)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        img_io = io.BytesIO()
        converted.save(img_io, 'JPEG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
