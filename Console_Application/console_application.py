'''
Some functions were modified from: https://www.kaggle.com/uds5501/cnn-segmentation-resnet-depth-5,
https://www.kaggle.com/drt2290078/mask-rcnn-sample-starter-code, and 
https://www.kaggle.com/ashishpatel26/chexnet-batch-normalization-hyparameter-tuning.
'''
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
import csv

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
    

# Make predictions on test images, write out sample submission 
def mask_rcnn_predict(image_fps, min_conf=0.95):   
    # assume square image
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    #resize_factor = ORIG_SIZE 
    out_str = ""
    with open(filepath, 'w') as file:
        for image_id in tqdm(image_fps): 
            ds = pydicom.read_file(image_id)
            image = ds.pixel_array
            # If grayscale. Convert to RGB for consistency.
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = np.stack((image,) * 3, -1) 
            image, window, scale, padding, crop = utils.resize_image(image, min_dim=config.IMAGE_MIN_DIM, min_scale=config.IMAGE_MIN_SCALE, max_dim=config.IMAGE_MAX_DIM, mode=config.IMAGE_RESIZE_MODE)
                
            patient_id = os.path.splitext(os.path.basename(image_id))[0]
    
            results = model.detect([image])
            r = results[0]
    
            #out_str = ""
            out_str += patient_id 
            out_str += ","
            assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
            if len(r['rois']) == 0: 
                pass
            else: 
                num_instances = len(r['rois'])
      
                for i in range(num_instances): 
                    if r['scores'][i] > min_conf: 
                        out_str += ' '
                        out_str += str(round(r['scores'][i], 2))
                        out_str += ' '
    
                        # x1, y1, width, height 
                        x1 = r['rois'][i][1]
                        y1 = r['rois'][i][0]
                        width = r['rois'][i][3] - x1 
                        height = r['rois'][i][2] - y1 
                        bboxes_str = "{} {} {} {}".format(x1*resize_factor, y1*resize_factor, \
                                                           width*resize_factor, height*resize_factor)   
                        out_str += bboxes_str

    return out_str
    

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
  
def find_patient_dcm_image(patient_id, test_image_directory):
    patient_test_data = []
    for filename in os.listdir(test_image_directory):
        if patient_id in filename:
            patient_test_data.append(filename)
        else:
            continue
    if len(patient_test_data) == 0:
        print("Unable to find dicom file for that patient.")
    return patient_test_data
    
def get_actual_bounding_box(patient_id):
    ground_truth_bounding_box = []
    with open('labels.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if (row['patientId'] == patient_id):
                try:
                    ground_truth_bounding_box.append(float(row['x']))
                    ground_truth_bounding_box.append(float(row['y']))
                    ground_truth_bounding_box.append(float(row['width']))
                    ground_truth_bounding_box.append(float(row['height']))
                except ValueError:
                  print("Pneumonia was not found in this patient. There is no ground truth bounding box.")

    return ground_truth_bounding_box

choice = ''
display_title_bar()
while choice != 'q':

    choice = get_user_choice()

    # Respond to the user's choice.
    display_title_bar()
    if choice == '1':
        # Load ResNet Model
        resnet_model = load_model("./resnet_model.json", "./resnet_model.h5")
        patient_id = get_patient_id()
        patient_test = find_patient_dcm_image(patient_id, "/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/")
        test_gen = Generator("/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/", patient_test, None, batch_size=20, image_size=256, shuffle=False, predict=True)
        model_predict(resnet_model, test_gen[0])
        gt_bb = get_actual_bounding_box(patient_id)
        pass
    elif choice == '2':
        # Load MaskRCNN
        mask_rcnn_model = load_model("./mask_rcnn_model.json", "./mask_rcnn_model.h5")
        patient_id = get_patient_id()
        patient_test = find_patient_dcm_image(patient_id, "/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/")
        mask_rcnn_predict(patient_test[1][0])
        pass
    elif choice == '3':
        # Load ChexNet
        chexnet_model = load_model("./chexnet_rcnn_model.json", "./chexnet_rcnn_model.h5")
        patient_id = get_patient_id()
        patient_test = find_patient_dcm_image(patient_id, "/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/")
        test_gen = Generator("/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/", patient_test, None, batch_size=25, image_size=256, shuffle=False, predict=True)
        model_predict(chexnet_model, test_gen[0])
        gt_bb = get_actual_bounding_box(patient_id)
        pass
    elif choice == 'q':
        print("\nExiting.")
    else:
        print("\nI didn't understand that choice.\n")
