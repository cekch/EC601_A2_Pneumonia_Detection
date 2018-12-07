import pydicom
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

def draw_bounding_boxes(box, im):
    rgb = np.array([255,0,0]).astype('int')
    im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)
    plt.imshow(im, cmap=pylab.cm.gist_gray)
    #plt.show()
    plt.imsave('./static/images/patient_plot.png', im)