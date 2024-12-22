from google.cloud import firestore
from google.oauth2 import service_account
import numpy as np
import face_recognition
import cv2
import requests
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image 
from flask_cors import CORS






credentials = service_account.Credentials.from_service_account_file(
    'face-1731f-firebase-adminsdk-0gs2t-650494e905.json'
)
firebase = firestore.Client(credentials=credentials)
collection_ref = firebase.collection('faces')

def get_name(img):

    try:
        rgb_image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img1 = face_recognition.face_encodings(rgb_image, face_recognition.face_locations(rgb_image) )[0]
        docs = collection_ref.stream()
        probs=[]
        name=[]


        for doc in docs:
            dic=doc.to_dict()
            img2=np.fromstring(dic['face'], dtype=float, sep=" ")

            if face_recognition.compare_faces([img1], img2):
                face_distance = face_recognition.face_distance([img1], img2)
                probs.append(1 - face_distance[0])
                name.append(dic['name'])
        

        return True,name[probs.index(max(probs))]
        
    except IndexError:
        return False,''

app = Flask(__name__)
CORS(app)  

@app.route('/get_name', methods=['POST'])
def get_name_endpoint():


    data = request.get_json()
    image_data = data['image']
    image_bytes = base64.b64decode(image_data)
    image =np.array(  Image.open(BytesIO(image_bytes)) )

    mes,name=get_name(image)
    s=''
    if mes:
        s=name
    else:
        s='0'

    return jsonify({'message': s}), 200

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000, use_reloader=False)