import numpy as np
import pydicom
import collections
import cv2
import matplotlib.pyplot as plt
import pydicom

#docker
def read_predicted(WORD_FILE):
    all_data = collections.defaultdict(list)
    with open(WORD_FILE, "r") as ins:
        for line in ins:
            line = line[:-1]
            line = line.split(",")
            list_bboxes = []
            if len(line) == 2:
                if line[1] != '' and line[1][0] == ' ':
                    line[1] = line[1][1:]
                value_split = line[1].split(" ")       
                #print(value_split)
                for index, value in enumerate(value_split):
                    bbox = []
                    if index % 5 == 0:
                        bbox.extend(value_split[index:index+5])
                    if bbox != []:
                        if bbox != [''] and bbox != ['PredictionString']:
                            list_bboxes.append(bbox)
            if line[0] != 'patientId':
                all_data[line[0]] = list_bboxes
            #print(list_bboxes)
            '''
            if len(line) == 2:
                #print(line[1])
                value_split = line[1].split(" ")
                
                value_split = value_split[1:]

                all_bboxes = []
                for index, value in enumerate(value_split):
                    bbox = []
                    if index % 5 == 0:
                        bbox.extend(value_split[index:index+5])
                    if bbox != []:
                        all_bboxes.append(bbox)
                    #print(all_bboxes)
            else:
                all_bboxes = []
            all_data[line[0]] = all_bboxes
            '''
    #for key in all_data:
    #    print(key, all_data[key])
    return all_data
def read_real(WORD_FILE):
    all_data = collections.defaultdict(list)
    with open(WORD_FILE, "r") as ins:
        for line in ins:
            line = line[:-1]
            value_split = line.split(",")
            patient_id = value_split[0]
            bbox = value_split[1:5]
            all_data[patient_id].append(bbox)
    return all_data


def iou():
    #print("Done")
    predicted_data = read_predicted("submission_resnet_new.csv")
    #print(len(predicted_data))
    real_data = read_real("stage_2_train_labels.csv")    
    all_iou = []
    normal_iou = []
    pneumonia_iou = []
    #print("Done")
    for patient_id in predicted_data:
        predictions = np.zeros((1024,1024))
        grounds = np.zeros((1024,1024))
        # print("Done")
        #predicted_bboxes = predicted_data['0e533087-5959-46e5-9fa5-95761a7d472f']
        #real_bboxes = real_data['0e533087-5959-46e5-9fa5-95761a7d472f']
        predicted_bboxes = predicted_data[patient_id]
        real_bboxes = real_data[patient_id]
        for bbox in predicted_bboxes:
            if bbox != []:
                x = int(float(bbox[1]))
                y = int(float(bbox[2]))
                w = int(float(bbox[3]))
                h = int(float(bbox[4]))
                predictions[x:(x+w), y:(y+h)] = 1
        for rbox in real_bboxes:
            if rbox[0] != '' and rbox[0] != 'x':
                x2 = int(float(rbox[0]))
                y2 = int(float(rbox[1]))
                w2 = int(float(rbox[2]))
                h2 = int(float(rbox[3]))
                grounds[x2:(x2+w2), y2:(y2+h2)] = 1
        intersect = np.logical_and(predictions, grounds)
        un = np.logical_or(predictions, grounds)
        intersection_count = sum(sum(intersect))
        union_count = sum(sum(un))
        unhealth = sum(sum(grounds))
        # plt.imshow(predictions)
        # plt.figure()
        # plt.imshow(grounds)
        # plt.figure()
        # plt.imshow(intersect)
        # plt.figure()
        # plt.imshow(un)
        # plt.figure()
        # plt.show()


        if unhealth > 0:
            pneumonia_iou.append(intersection_count/union_count)
            all_iou.append(intersection_count/union_count)
        else:
            if union_count > 0:
                normal_iou.append(intersection_count/union_count)
                all_iou.append(intersection_count/union_count)
            else:
                normal_iou.append(1.0)
                all_iou.append(1.0)

        # image2 = image0 + image1
        # union_count = 0
        # intersection_count = 0
        # combined = image2.flatten()
        # for pixel in combined:
        #     if pixel == 1:
        #         union_count += 1
        #     if pixel == 2:
        #         union_count += 1
        #         intersection_count += 1
        # if union_count != 0:
        #     image_iou = intersection_count/union_count
        # else:
        #     image_iou = 1
        # all_iou.append(image_iou)
    #print(pneumonia_iou)
    print("Overall:")
    print(sum(all_iou) / float(len(all_iou)))
    print("# of Normals:")
    print(len(normal_iou))
    print("Normal CXRs:")
    print(sum(normal_iou) / float(len(normal_iou)))
    print("# of Pneumonia:")
    print(len(pneumonia_iou))
    print("Pneumonia CXRs:")
    print(sum(pneumonia_iou) / float(len(pneumonia_iou)))


    #for patient_id, bboxes in predicted_data.items():

def show_compare():
    predicted_data = read_predicted("1.csv")
    real_data = read_real("stage_2_train_labels.csv")
    patient_id = '0fda72a2-f383-4f69-af8e-e16a0fbac621'

    predicted_bboxes = predicted_data[patient_id]
    real_bboxes = real_data[patient_id]

    image_id = "stage_2_train_images/%s.dcm" % patient_id
    print(image_id)
    ds = pydicom.read_file(image_id)
    image = ds.pixel_array
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1) 
    for bbox in predicted_bboxes:
        if bbox != []:
            x = int(float(bbox[0]))
            y = int(float(bbox[1]))
            w = int(float(bbox[2]))
            h = int(float(bbox[3]))
            cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 3, 1)
    for rbox in real_bboxes:
        if rbox[0] != '' and rbox[0] != 'x':
            x2 = int(float(rbox[0]))
            y2 = int(float(rbox[1]))
            w2 = int(float(rbox[2]))
            h2 = int(float(rbox[3]))
            cv2.rectangle(image, (x2, y2), (w2, h2), (0, 255, 0), 3, 1)

    plt.figure() 
    plt.imshow(image, cmap=plt.cm.gist_gray)
    plt.show()    

iou()
#read_predicted("1.csv")
#read_real("stage_2_train_labels.csv")