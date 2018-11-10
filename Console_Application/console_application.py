import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from sklearn.utils import shuffle
import os
import pydicom
from generator import Generator
from skimage import measure
from skimage.transform import resize

def display_title_bar():
    # Clears the terminal screen, and displays a title bar.
    os.system('clear')

    print("\t**********************************************")
    print("\t***  Predictor - Detect Lung Opacities!  ***")
    print("\t**********************************************")

def get_user_choice():
    # Let users know what they can do.
    print("\n[1] ResNet")
    print("[2] MaskRCNN")
    print("[3] ChexNet")
    print("[q] Quit.")

    return input("Which model would you like to use? ")
    
def get_patient_id():    
    return input("What is the patient id? ")
    
def load_model(model_json, model_h5):
    # load json and create model
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_h5)
    print("Loaded model from disk")
    
    return loaded_model
    
def resnet_predict(model, test_image_info):
    filename = test_image_info[1][0]
    preds = model.predict(test_image_info[0])
    pred = resize(preds[0], (1024, 1024), mode='reflect')
    print(type(pred[0]))
    print(len(pred[0]))
    print(pred[0])
    comp = pred[:, :, 0] > 0.5
    comp = measure.label(comp)
    predictionString = ''
    for region in measure.regionprops(comp):
        # retrieve x, y, height and width
        y, x, y2, x2 = region.bbox
        height = y2 - y
        width = x2 - x
        # proxy for confidence score
        conf = np.mean(pred[y:y+height, x:x+width])
        # add to predictionString
        predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' '
        # add filename and predictionString to dictionary
        filename = filename.split('.')[0]
        #submission_dict[filename] = predictionString
    print("prediction_string =" + str(predictionString))
    return predictionString
    

# define iou or jaccard loss function
def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score

# combine bce loss and iou loss
def iou_bce_loss(y_true, y_pred):
    return 0.4 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.6 * iou_loss(y_true, y_pred)

def evalulate_model(X, Y, loaded_model, loss_type, optimizer_type):
    loaded_model.compile(loss=loss_type, optimizer=optimzer_type, metrics=['accuracy'])
    score = loaded_model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
  
def find_patient_dcm_image(patient_id, test_image_directory):
    patient_test_data = []
    for filename in os.listdir(test_image_directory):
        if patient_id in filename:
            patient_test_data.append(filename)
        else:
            continue
    return patient_test_data

choice = ''
display_title_bar()
while choice != 'q':

    choice = get_user_choice()

    # Respond to the user's choice.
    display_title_bar()
    if choice == '1':
        # Load ResNet Model
        resnet_model = load_model("./model.json", "./model.h5")
        patient_id = get_patient_id()
        patient_test = find_patient_dcm_image(patient_id, "/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/")
        test_gen = Generator("/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/", patient_test, None, batch_size=20, image_size=256, shuffle=False, predict=True)
        resnet_predict(resnet_model, test_gen[0])
        #evaluate_model(resnet_model, X, Y, iou_bce_loss, )
        pass
    elif choice == '2':
        # Load MaskRCNN
        pass
    elif choice == '3':
        # Load ChexNet
        pass
    elif choice == 'q':
        print("\nBye.")
    else:
        print("\nI didn't understand that choice.\n")
