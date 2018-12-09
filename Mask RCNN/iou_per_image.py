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
            if len(line) == 2:
                print(line[1])
                value_split = line[1].split(" ")
                
                value_split = value_split[1:]

                all_bboxes = []
                for index, value in enumerate(value_split):
                    bbox = []
                    if index % 5 == 0:
                        bbox.extend(value_split[index+1:index+5])
                    if bbox != []:
                        all_bboxes.append(bbox)
                    #print(all_bboxes)
            else:
                all_bboxes = []
            all_data[line[0]] = all_bboxes
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
    print("Done")
    predicted_data = read_predicted("submission_chexnet 2.csv")
    real_data = read_real("stage_2_train_labels.csv")
    all_iou = []
    print("Done")
    for patient_id in predicted_data:
        #print(patient_id)
        image0 = np.zeros((1024,1024))
        image1 = np.zeros((1024,1024))
        # print("Done")
        # #predicted_bboxes = predicted_data['23ca0450-4138-4e7a-9489-0f5b6a91031d']
        # #real_bboxes = real_data['23ca0450-4138-4e7a-9489-0f5b6a91031d']
        predicted_bboxes = predicted_data[patient_id]
        real_bboxes = real_data[patient_id]
        #print(predicted_bboxes)
        #print(real_bboxes)
        for bbox in predicted_bboxes:
            if bbox != []:
                print(bbox)
                x = int(float(bbox[0]))
                y = int(float(bbox[1]))
                w = int(float(bbox[2]))
                h = int(float(bbox[3]))
                image0[x:(x+w), y:(y+h)] = 1
        for rbox in real_bboxes:
            if rbox[0] != '' and rbox[0] != 'x':
                x2 = int(float(rbox[0]))
                y2 = int(float(rbox[1]))
                w2 = int(float(rbox[2]))
                h2 = int(float(rbox[3]))
                image1[x2:(x2+w2), y2:(y2+h2)] = 1
        image2 = np.logical_and(image0,image1)
        image3 = np.logical_or(image0,image1)
        intersection_count = sum(sum(image2))
        union_count = sum(sum(image3))
        if union_count != 0:
            all_iou.append(intersection_count/union_count)
        else:
            all_iou.append(1.0)
        #plt.imshow(image2)
        #plt.show()
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
    print(sum(all_iou) / float(len(all_iou)))


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