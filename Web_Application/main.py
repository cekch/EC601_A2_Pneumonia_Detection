from flask import Flask, render_template, request
from werkzeug import secure_filename  
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
		return 'file uploaded successfully'

@app.route("/blank")
def blank():
	return "Hello, Blank!"

if __name__ == "__main__":
	app.run(debug=True)