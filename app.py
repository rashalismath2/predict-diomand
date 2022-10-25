from flask import Flask
from flask_restful import Resource, Api, reqparse
import werkzeug
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tempfile

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

        img = image.load_img(ofname, target_size = (220,220))
   
        X = image.img_to_array(img)
        X = np.expand_dims(X, axis =0)
        images = np.vstack([X])


        # color detection
        CLASSES=['Blue', 'Pink', 'Purple', 'Red', 'Yellow']
        predict_x=color_model.predict(images)[0]
        pred_class=np.argmax(predict_x,axis=0)
        
        predicted_color = '{}'.format(CLASSES[pred_class]) 

        # cutshape
        val = model.predict(images)[0]
        val=np.argmax(val,axis=0)
        print(val)
        cutshapereturn=""
        if val == 0:
            cutshapereturn="emerald"
        elif val == 1:
            cutshapereturn="heart"
        elif val == 2:    
            cutshapereturn="marquish"
        elif val == 3:
            cutshapereturn="oval"
        else:
            cutshapereturn="oval"
        return {
            "cutshape":cutshapereturn,
            "color":predicted_color
        }
api.add_resource(SaveImage, '/predict')

if __name__ == '__main__':
    app.run(debug=True)


