from flask import Flask, render_template, request
from werkzeug import secure_filename
import pydicom
import numpy as np
from PIL import Image
import pylab
import matplotlib.pyplot as plt
import draw_bounding_boxes as draw_bb
app = Flask(__name__)

@app.route("/")
def home():
	return render_template("home.html")
	
@app.route("/patient_plot")
def display_bbox(x, y, width, height, im):
    box = [x,y,width,height]
    draw_bb.draw_bounding_boxes(box, im) 

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


@app.route("/blank")
def blank():
	return "Hello, Blank!"

if __name__ == "__main__":
	app.run(debug=True)