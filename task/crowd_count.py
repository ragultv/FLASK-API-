from flask import Flask, request, send_file
from PIL import Image
import numpy as np
import cv2
import io
import tempfile
import os
from lwcc import LWCC
app = Flask(__name__)


def crowd_count(img_np):
    # Placeholder for the actual crowd counting function
    # Assuming it returns a count value directly
    results = LWCC.get_count(img_np)
    return results


@app.route('/count', methods=['POST'])
def count():
    if request.method == 'POST':
        image = request.files.get('file')
        # Open the image and convert to NumPy array
        img1 = Image.open(image)
        img1_np = np.array(img1)  # Convert PIL Image to NumPy array

        # Get the crowd count
        results = crowd_count(img1_np)

        # Add text to the original image (adjust position, font, etc. as needed)
        img1_np_bgr = cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR)
        cv2.putText(img1_np_bgr, f"Count: {int(results)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the modified NumPy array back to a PIL image
        result_img_pil = Image.fromarray(cv2.cvtColor(img1_np_bgr, cv2.COLOR_BGR2RGB))
        img_io = io.BytesIO()
        result_img_pil.save(img_io, 'JPEG')
        img_io.seek(0)
        # Return the image with predictions
        return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
