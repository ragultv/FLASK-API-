
from ultralytics import YOLO
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

model = YOLO(r"C:\Users\tragu\Downloads\best.pt")

app = Flask('__name__')


def predict(img):
    results = model.predict(img)


@app.route('/predict', methods=['POST'])
def predict_route():
    if request.method == 'POST':
        img = request.files.get('file')
        converted = Image.open(img)
        imgarr = np.array(converted)
        licence_plate = predict(imgarr)
        # Example of processing the results (assuming the model outputs bounding boxes and labels)
        result_data = []
        for detection in licence_plate.xyxy:
            bbox = detection[:4]  # bounding box coordinates
            confidence = detection[4]  # confidence score
            label = detection[5]  # class label
            result_data.append({
                'bbox': bbox.tolist(),
                'confidence': confidence.item(),
                'label': label.item()
            })

        result = {'messsage': 'licence plate detected'}
        return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)

