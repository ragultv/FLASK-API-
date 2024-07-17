from flask import Flask, request,jsonify
from retinaface import RetinaFace
import numpy as np
from PIL import Image
def detect_faces(image):
    rfs=RetinaFace.detect_faces(image)

app= Flask('__name__')

@app.route('/detect',methods=['POST'])
def detect():
    if request.method =='POST':
        image=request.files.get('file')
        converted=Image.open(image)
        imgarr=np.array(converted)
        faces=detect_faces(imgarr)
        result = {'message': 'Detection complete'}
        return jsonify(result)
if __name__ == '__main__':
    app.run(debug=True)


