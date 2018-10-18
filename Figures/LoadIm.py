import pydicom
import numpy as np
from mrcnn import utils
import matplotlib.pyplot as plt
import cv2

def visualize(): 
    image_id = "stage_1_test_images/045e3b0d-46ef-4a89-8795-0a85a2be0d6b.dcm"
    ds = pydicom.read_file(image_id)
    
    # original image 
    image = ds.pixel_array
    
    # assume square image 
    #resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    
    # If grayscale. Convert to RGB for consistency.
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1) 

    #Eric
    cv2.rectangle(image, (216, 376), ((216+244), (376+372)), (77, 255, 9), 3, 1)
    cv2.rectangle(image, (628, 352), ((628+248), (352+352)), (77, 255, 9), 3, 1)
    cv2.rectangle(image, (260, 100), ((260+200), (100+344)), (77, 255, 9), 3, 1)
    #Caroline
    cv2.rectangle(image, (960, 320), (960+64, 320+128), (255, 10 , 10), 3, 1)
    cv2.rectangle(image, (128, 640), ((128+192), (640+384)), (255, 10, 10), 3, 1)
    #Sarthak
    cv2.rectangle(image, (638, 192), ((638+193), (192+447)), (10, 10, 255), 3, 1)
    cv2.rectangle(image, (256, 256), ((256+192),(256+510)), (10, 10, 255), 3, 1)
    plt.figure() 
    plt.imshow(image, cmap=plt.cm.gist_gray)
    plt.show()

    image_id = "stage_1_test_images/c18d1138-ba74-4af5-af21-bdd4d2c96bb5.dcm"
    ds = pydicom.read_file(image_id)
    
    # original image 
    image = ds.pixel_array
    
    # assume square image 
    #resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    
    # If grayscale. Convert to RGB for consistency.
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1) 
    #Eric
    cv2.rectangle(image, (244, 404), ((244+216), (404+340)), (77, 255, 9), 3, 1)
    #Caroline
    cv2.rectangle(image, (192, 384), (192+256, 384+640), (255, 10 , 10), 3, 1)
    #Sarthak
    cv2.rectangle(image, (577, 191), ((577+254),(191+448)), (10, 10, 255), 3, 1)
    cv2.rectangle(image, (258, 384), ((258+189),(384+257)), (10, 10, 255), 3, 1)

    plt.figure() 
    plt.imshow(image, cmap=plt.cm.gist_gray)
    plt.show()    


visualize()