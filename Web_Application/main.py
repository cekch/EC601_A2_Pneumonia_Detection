import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug import secure_filename
import pydicom
import numpy as np
from PIL import Image
import pylab
import matplotlib.pyplot as plt
import draw_bounding_boxes as draw_bb
import load_and_predict as lp
app = Flask(__name__)

#resnet_model = lp.load_machine_learning_model("./model_files/resnet_model.json", "./model_files/resnet_model.h5", '1')
mask_rcnn_model = lp.load_machine_learning_model("", "./model_files/mask_rcnn_model.h5", '2')
chexnet_model = lp.load_machine_learning_model("./model_files/chexnet_model.json", "./model_files/chexnet_model.h5", '3')
global graph
graph = tf.get_default_graph()


@app.route("/")
def home():
	return render_template("home.html")

@app.route("/patient_plot")
def display_bbox(boxes, im, patient_name):
	# box = [x,y,width,height]
	draw_bb.draw_bounding_boxes(boxes, im, patient_name)

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
		patient_id = f.filename.split('.')[0]
		files = []
		files.append(f.filename)
		testgen = lp.test_gen(files)
		preds = []
		with graph.as_default():
			#pred_string1 = lp.model_predict(resnet_model, testgen[0])
			pred_string2 = lp.mask_rcnn_predict(mask_rcnn_model, "./" + f.filename, patient_id)
			pred_string3 = lp.model_predict(chexnet_model, testgen[0])
			#preds.append(pred_string1)
			preds.append(pred_string2)
			preds.append(pred_string3)

		predboxes=[]
		for pred in preds:
			try:
				pred_list = pred.split()
				pred_len = len(pred_list)
				num_boxes = pred_len/5
				i=0
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
			except:
				pass

		patient_name = f.filename[:-4]
		pic_name = '/static/images/%s.png' % patient_name
		display_bbox(predboxes, im, patient_name)
		# x = 640
		# y = 450
		# width = 65
		# height = 188
		#return 'file uploaded successfully'
	return render_template('patient_plot.html', name = 'patient plot', url = pic_name)

@app.route("/blank")
def blank():
	return "Hello, Blank!"

if __name__ == "__main__":
	app.run(debug=True)
