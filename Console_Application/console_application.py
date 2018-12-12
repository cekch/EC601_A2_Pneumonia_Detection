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
from keras.models import load_model
import pylab
import model as modellib
import config
from config import Config
import utils
from plot_loss import plot_stats
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

ROOT_DIR = '/project/ece601/A2_Pneumonia_Detection_MaskRCNN/'

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
    print("[4] Plot Model Metrics")
    print("[5] Ensemble")
    print("[q] Quit.")

    return input("Which model would you like to use? ")

def get_patient_id():
    return input("What is the patient id? ")

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
    if (out_str == ''):
        print("No lung opacities found.")
    else:
        print("Mask-RCNN Prediction String = " + out_str)
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

    print(ground_truth_bounding_box)
    return ground_truth_bounding_box

def draw(pred_string, ground_truth, patient_id):
    """
    Method to draw single patient with bounding box(es) if present

    """
    # --- Open DICOM file
    d = pydicom.read_file('/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/%s.dcm' % patient_id)
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # --- Find Predicted boxes
    pred_list = pred_string.split()
    pred_len = len(pred_list)
    num_boxes = pred_len/5
    i=0
    predboxes=[]
    while(num_boxes):
        conf = pred_list[i]
        xmin = pred_list[i+1]
        ymin = pred_list[i+2]
        wid = pred_list[i+3]
        ht = pred_list[i+4]
        box = [conf,xmin,ymin,wid,ht]
        predboxes.append(box)
        i=i+5
        num_boxes=num_boxes-1

    gt_len = len(ground_truth)
    num_gt_boxes = gt_len/4
    j=0
    gtboxes=[]
    while(num_gt_boxes):
        xmin = ground_truth[j]
        ymin = ground_truth[j+1]
        wid = ground_truth[j+2]
        ht = ground_truth[j+3]
        box = [xmin,ymin,wid,ht]
        gtboxes.append(box)
        j=j+4
        num_gt_boxes=num_gt_boxes-1

    # --- Add boxes in red if present
    for box in predboxes:
        # rgb = np.floor(np.random.rand(3) * 256).astype('int')
        rgb = np.array([255,0,0]).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    for box in gtboxes:
        # rgb = np.floor(np.random.rand(3) * 256).astype('int')
        rgb = np.array([0,255,0]).astype('int')
        im = overlay_gtbox(im=im, box=box, rgb=rgb, stroke=6)

    plt.imshow(im, cmap=pylab.cm.gist_gray)
    # pyla.axis('off')
    # matplotlib.pyplot.show()
    plt.show()

def overlay_box(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    conf = box[0]
    box = [int(float(b)) for b in box[1:]]

    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    pilimg = Image.fromarray(im)
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype('Roboto-Bold.ttf', size=45)
    (x, y) = ((x1+x2)/2, y1-46)
    message = "conf: "+str(conf)[:5]
    color = 'rgb(255, 255, 0)' # black color
    draw.text((x, y), message, fill=color, font=font)
    # im = np.array(draw)

    im = np.array(pilimg)

    return im

def overlay_gtbox(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    box = [int(float(b)) for b in box]

    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im

def draw_all(pred_string1, pred_string2, pred_string3, ground_truth, patient_id):
    """
    Method to draw single patient with bounding box(es) if present

    """
    # --- Open DICOM file
    d = pydicom.read_file('/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/%s.dcm' % patient_id)
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # --- Find Predicted boxes
    pred_list1 = pred_string1.split()
    pred_len1 = len(pred_list1)
    num_boxes1 = pred_len1/5
    i=0
    predboxes1=[]
    while(num_boxes1):
        conf = pred_list1[i]
        xmin = pred_list1[i+1]
        ymin = pred_list1[i+2]
        wid = pred_list1[i+3]
        ht = pred_list1[i+4]
        box = [conf, xmin,ymin,wid,ht]
        predboxes1.append(box)
        i=i+5
        num_boxes1=num_boxes1-1

    pred_list2 = pred_string2.split()
    pred_len2 = len(pred_list2)
    num_boxes2 = pred_len2/5
    j=0
    predboxes2=[]
    while(num_boxes2):
        conf = pred_list2[j]
        xmin = pred_list2[j+1]
        ymin = pred_list2[j+2]
        wid = pred_list2[j+3]
        ht = pred_list2[j+4]
        box = [conf, xmin,ymin,wid,ht]
        predboxes2.append(box)
        j=j+5
        num_boxes2=num_boxes2-1

    pred_list3 = pred_string3.split()
    pred_len3 = len(pred_list3)
    num_boxes3 = pred_len3/5
    k=0
    predboxes3=[]
    while(num_boxes3):
        conf = pred_list3[k]
        xmin = pred_list3[k+1]
        ymin = pred_list3[k+2]
        wid = pred_list3[k+3]
        ht = pred_list3[k+4]
        box = [conf, xmin,ymin,wid,ht]
        predboxes3.append(box)
        k=k+5
        num_boxes3=num_boxes3-1

    gt_len = len(ground_truth)
    num_gt_boxes = gt_len/4
    j=0
    gtboxes=[]
    while(num_gt_boxes):
        xmin = ground_truth[j]
        ymin = ground_truth[j+1]
        wid = ground_truth[j+2]
        ht = ground_truth[j+3]
        box = [xmin,ymin,wid,ht]
        gtboxes.append(box)
        j=j+4
        num_gt_boxes=num_gt_boxes-1

    # --- Add boxes with random color if present
    for box in predboxes1:
        # rgb = np.floor(np.random.rand(3) * 256).astype('int')
        rgb = np.array([255,255,0]).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    for box in predboxes2:
        # rgb = np.floor(np.random.rand(3) * 256).astype('int')
        rgb = np.array([0,255,255]).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    for box in predboxes3:
        # rgb = np.floor(np.random.rand(3) * 256).astype('int')
        rgb = np.array([255,0,255]).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    for box in gtboxes:
        # rgb = np.floor(np.random.rand(3) * 256).astype('int')
        rgb = np.array([0,255,0]).astype('int')
        im = overlay_gtbox(im=im, box=box, rgb=rgb, stroke=6)

    plt.imshow(im, cmap=pylab.cm.gist_gray)
    # pyla.axis('off')
    # matplotlib.pyplot.show()
    plt.show()

#Load all models at start up
resnet_model = load_machine_learning_model("./resnet_model.json", "./resnet_model.h5", '1')
mask_rcnn_model = load_machine_learning_model("", "./mask_rcnn_model.h5", '2')
chexnet_model = load_machine_learning_model("./chexnet_model.json", "./chexnet_model.h5", '3')


choice = ''
display_title_bar()
while choice != 'q':
    choice = get_user_choice()
    # Respond to the user's choice.
    display_title_bar()
    if choice == '1':
        #ResNet Model
        patient_id = get_patient_id()
        patient_test = find_patient_dcm_image(patient_id, "/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/")
        test_gen = Generator("/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/", patient_test, None, batch_size=20, image_size=256, shuffle=False, predict=True)
        pred_string = model_predict(resnet_model, test_gen[0])
        gt_bb = get_actual_bounding_box(patient_id)
        draw(pred_string,gt_bb,patient_id)
        print("Red = ResNet Predictions")
        print("Green = Ground Truth")
        print
        pass
    elif choice == '2':
        #MaskRCNN Model
        patient_id = get_patient_id()
        patient_test = find_patient_dcm_image(patient_id, "/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/")
        #test_gen = Generator("/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/", patient_test, None, batch_size=8, image_size=256, shuffle=False, predict=True)
        pred_string = mask_rcnn_predict(mask_rcnn_model, "/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/" + patient_test[0], patient_id)
        print("Red = Mask-RCNN Predictions")
        print("Green = Ground Truth")
        gt_bb = get_actual_bounding_box(patient_id)
        draw(pred_string,gt_bb,patient_id)
        pass
    elif choice == '3':
        #ChexNet Model
        patient_id = get_patient_id()
        patient_test = find_patient_dcm_image(patient_id, "/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/")
        test_gen = Generator("/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/", patient_test, None, batch_size=25, image_size=256, shuffle=False, predict=True)
        pred_string = model_predict(chexnet_model, test_gen[0])
        gt_bb = get_actual_bounding_box(patient_id)
        print("Red = ChexNet Predictions")
        print("Green = Ground Truth")
        draw(pred_string,gt_bb,patient_id)
        pass
    elif choice == '4':
        plot_stats()
    elif choice == '5':
        #Ensemble Model
        patient_id = get_patient_id()
        patient_test = find_patient_dcm_image(patient_id, "/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/")
        test_gen = Generator("/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/", patient_test, None, batch_size=25, image_size=256, shuffle=False, predict=True)
        pred_string1 = model_predict(resnet_model, test_gen[0])
        pred_string2 = mask_rcnn_predict(mask_rcnn_model, "/project/ece601/A2_Pneumonia_Detection/Dataset/stage_1_test_images/" + patient_test[0], patient_id)
        pred_string3 = model_predict(chexnet_model, test_gen[0])
        gt_bb = get_actual_bounding_box(patient_id)
        print("Yellow = ResNet Predictions")
        print("Blue = Mask-RCNN Predictions")
        print("Purple = ChexNet Predictions")
        print("Green = Ground Truth")
        draw_all(pred_string1, pred_string2, pred_string3, gt_bb, patient_id)
    elif choice == 'q':
        print("\nExiting.")
    else:
        print("\nI didn't understand that choice.\n")
