from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

def model_predict(model, test_image_info):
    filename = test_image_info[1][0]
    preds = model.predict(test_image_info[0])
    pred = resize(preds[0], (1024, 1024), mode='reflect')
    comp = pred[:, :, 0] > 0.5
    comp = measure.label(comp)
    prediction_string = ''
    for region in measure.regionprops(comp):
        # retrieve x, y, height and width
        y, x, y2, x2 = region.bbox
        height = y2 - y
        width = x2 - x
        # proxy for confidence score
        conf = np.mean(pred[y:y+height, x:x+width])
        # add to predictionString
        prediction_string += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' '
        # add filename and predictionString to dictionary
        filename = filename.split('.')[0]
        #submission_dict[filename] = predictionString
    if (prediction_string == ''):
        print("No lung opacties found.")
    else:
      print("bounding boxes = " + str(prediction_string))
    return prediction_string

app = Flask(__name__)
api = Api(app)

json_file = open("model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictOpacity(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        # vectorize the user's query and make a prediction
        pred_string = model_predict(loaded_model,user_query)

        if pred_string == '':
            pred_text = 'No Opacities Found'
        else:
            pred_text = pred_string

        # create JSON object
        output = {'prediction': pred_text}

        return output


api.add_resource(PredictOpacity, '/')

if __name__ == '__main__':
    app.run(debug=True)
