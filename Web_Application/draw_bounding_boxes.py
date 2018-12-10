import pydicom
import numpy as np
from PIL import Image
import pylab
import matplotlib.pyplot as plt

def overlay_box(im, box, rgb, stroke=1):
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

# def draw_bounding_boxes(pred_string, im, patient_id):
#     pred_list = pred_string.split()
#     pred_len = len(pred_list)
#     num_boxes = pred_len/5
#     i=0
#     predboxes=[]
#     while(num_boxes):
#         conf = pred_list[i]
#         xmin = pred_list[i+1]
#         ymin = pred_list[i+2]
#         wid = pred_list[i+3]
#         ht = pred_list[i+4]
#         box = [xmin,ymin,wid,ht]
#         predboxes.append(box)
#         i=i+5
#         num_boxes=num_boxes-1
#     for box in predboxes:
#         rgb = np.array([255,0,0]).astype('int')
#         im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)
#     plt.imshow(im, cmap=pylab.cm.gist_gray)
#     plt.imsave('./static/images/%s.png' % patient_id, im)

def draw_bounding_boxes(boxes, im, patient_name):
    for box in boxes:
        rgb = np.array([255,0,0]).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)
    plt.imshow(im, cmap=pylab.cm.gist_gray)
    #plt.show()
    plt.imsave('./static/images/%s.png' % patient_name, im)
