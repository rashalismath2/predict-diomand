from flask import Flask
from flask_restful import Resource, Api, reqparse
import werkzeug
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sn

import cv2
from random import randint

import numpy as np

def edge_and_cut(img):
    try:
        img_w, img_h = 220, 220
        edges = cv2.Canny(img, img_w, img_h)

        if(np.count_nonzero(edges) > edges.size/10000):
            pts = np.argwhere(edges > 0)
            y1, x1 = pts.min(axis=0)
            y2, x2 = pts.max(axis=0)

            new_img = img[y1:y2, x1:x2]           # crop the region
            new_img = cv2.resize(new_img, (img_w, img_h))  # Convert back
        else:
            new_img = cv2.resize(img, (img_w, img_h))

    except Exception as e:
        print(e)
        new_img = cv2.resize(img, (img_w, img_h))

    return new_img

def crop_images(Imgs):
    img_w, img_h = 220, 220
    CroppedImages = np.ndarray(shape=(len(Imgs), img_w, img_h, 3), dtype=np.int)

    ind = 0
    for im in Imgs:
        x = edge_and_cut(im)
        CroppedImages[ind] = x
        cv2.imwrite("img{0}.jpg".format(ind), im)
        ind += 1

    return CroppedImages

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('file',
                    type=werkzeug.datastructures.FileStorage,
                    location='files',
                    required=True,
                    help='provide a file')


class SaveImage(Resource):

    def post(self):
        args = parser.parse_args()
        # read like a stream
        stream = args['file']
        ofile, ofname = tempfile.mkstemp()
        stream.save(ofname)

        model = tf.keras.models.load_model("./model_gemstones.h5")
        color_model = tf.keras.models.load_model("./model_colordetection.h5")

        img = image.load_img(ofname, target_size = (220, 220))

        X = image.img_to_array(img)
        X = np.expand_dims(X, axis=0)
        images = np.vstack([X])

        # color detection
        img = cv2.imread(ofname)    
        img_w, img_h = 220, 220
        cropped_img = cv2.resize(img,(int(img_w*1.5), int(img_h*1.5)))       # resize the image (images are different sizes)
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV) # converts an image from BGR color space to HSV
        cropped_img=crop_images([cropped_img])[0]
        cropped_img_X = image.img_to_array(cropped_img)
        cropped_img_X = np.expand_dims(cropped_img_X, axis=0)
        cropped_img_final = np.vstack([cropped_img_X])

        CLASSES = ['Blue', 'Pink', 'Purple', 'Red', 'Yellow']
        predict_x = color_model.predict(cropped_img_final).reshape(5)
        pred_class = np.argmax(predict_x,axis=0)

        predicted_color = '{}'.format(CLASSES[pred_class])

        # cutshape
        val = model.predict(images)[0]
        val = np.argmax(val,axis=0)
        print(val)
        cutshapereturn = ""
        if val == 0:
            cutshapereturn = "emerald"
        elif val == 1:
            cutshapereturn = "heart"
        elif val == 2:
            cutshapereturn = "marquish"
        elif val == 3:
            cutshapereturn = "oval"
        else:
            cutshapereturn = "oval"
        return {
            "cutshape": cutshapereturn,
            "color": predicted_color
        }

api.add_resource(SaveImage, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
