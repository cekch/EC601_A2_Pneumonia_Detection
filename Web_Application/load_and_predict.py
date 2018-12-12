import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.utils import shuffle
import os
import pydicom
from skimage import measure
from skimage.transform import resize
import csv
import sys
sys.path.insert(0, './model_files')
import model as modellib
import config
from config import Config
from generator import Generator
import utils


ROOT_DIR = './model_files/Mask_RCNN/'

class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """
    # Give the configuration a recognizable name
    NAME = 'pneumonia'
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    BACKBONE = 'resnet50'

    NUM_CLASSES = 2  # background + 1 pneumonia classes

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    #RPN_ANCHOR_SCALES = (32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 3
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_NMS_THRESHOLD = 0.1

    STEPS_PER_EPOCH = 100

class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def load_machine_learning_model(model_json, model_h5, model_type):
    if model_type == '1' or model_type == '3':
        # load json and create model
        json_file = open(model_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(model_h5)
    elif model_type == '2':
        inference_config = InferenceConfig()
        loaded_model = modellib.MaskRCNN(mode='inference', config=inference_config, model_dir=ROOT_DIR)
        loaded_model.load_weights(model_h5, by_name=True)

    print("Loaded ",model_type ," model from disk")

    return loaded_model

# resnet_model = load_machine_learning_model("./model_files/resnet_model.json", "./model_files/resnet_model.h5", '1')
# mask_rcnn_model = load_machine_learning_model("", "./model_files/mask_rcnn_model.h5", '2')
# chexnet_model = load_machine_learning_model("./model_files/chexnet_model.json", "./model_files/chexnet_model.h5", '3')

def test_gen(filename):
    return Generator("./", filename, None, batch_size=25, image_size=256, shuffle=False, predict=True)

# Make predictions on test images, write out sample submission
def mask_rcnn_predict(model, image_fps, patient_id, min_conf=0.95):
    # assume square image
    resize_factor = 1024 / model.config.IMAGE_SHAPE[0]
    #resize_factor = ORIG_SIZE
    out_str = ""
    #for image_id in tqdm(image_fps):
    ds = pydicom.read_file(image_fps)
    image = ds.pixel_array
    # If grayscale. Convert to RGB for consistency.
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1)
    image, window, scale, padding, crop = utils.resize_image(image, min_dim=model.config.IMAGE_MIN_DIM, min_scale=model.config.IMAGE_MIN_SCALE, max_dim=model.config.IMAGE_MAX_DIM, mode=model.config.IMAGE_RESIZE_MODE)
    #patient_id = os.path.splitext(os.path.basename(image_id))[0]

    results = model.detect([image])
    r = results[0]

    #out_str = ""
    #out_str += patient_id
    #out_str += ","
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
                bboxes_str = "{} {} {} {}".format(str(x1*resize_factor), str(y1*resize_factor), \
                                                   str(width*resize_factor), str(height*resize_factor))
                out_str += bboxes_str
    print("Mask-RCNN Prediction String = " + out_str)
    return out_str

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
