from flask import Flask, render_template, request
from werkzeug import secure_filename
import pydicom
import numpy as np
from PIL import Image
import pylab
import matplotlib.pyplot as plt
#import base64
#import plot_bounding_boxes
app = Flask(__name__)

@app.route("/")
def home():
	return render_template("home.html")

@app.route("/about")
def about():
	return render_template("about.html")

@app.route("/upload")
def upload_file():
	return render_template("upload.html")

@app.route("/uploader", methods = ['GET', 'POST'])
def upload_file2():
	if request.method == 'POST':
		f = request.files['file']
		f.save(secure_filename(f.filename))
		d = pydicom.read_file(f.filename)
		im = d.pixel_array
		im = np.stack([im] * 3, axis=2)
		x = 640
		y = 450
		width = 65
		height = 188
		display_bbox(x, y, width, height, im)
		#return 'file uploaded successfully'
        return render_template('patient_plot.html', name = 'patient plot', url = '/static/images/patient_plot.png')

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
	
@app.route("/patient_plot")
def display_bbox(x, y, width, height, im):
    rgb = np.array([255,0,0]).astype('int')
    box = [x,y,width,height]
    im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)
    plt.imshow(im, cmap=pylab.cm.gist_gray)
    #plt.show()
    plt.imsave('./static/images/patient_plot.png', im)

@app.route("/blank")
def blank():
	return "Hello, Blank!"

if __name__ == "__main__":
	app.run(debug=True)