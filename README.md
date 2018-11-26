# EC601_A2_Pneumonia_Detection

##Objective
The goal of this project is to develop a product that detects pneumonia in Chest X-Rays (CXRs) by detecting opacities in the lungs. This product is designed for medical professionals to use by inputting CXRs into the product, and determine based on results whether or not a patient has pneumonia.

To achieve this objective, we will need to research, customize, train and test machine learning models in order to automatically detect areas of pneumonia in CXRs. We are currently trying out multiple different convolutional neural networks (CNNs) to use for this objective. Some CNNs that were commonly used for the Pneumonia Kaggle challenge:
'''
ResNet: https://www.kaggle.com/uds5501/cnn-segmentation-resnet-depth-5
Mask-RCNN: https://www.kaggle.com/drt2290078/mask-rcnn-sample-starter-code
ChexNet: https://www.kaggle.com/ashishpatel26/chexnet-batch-normalization-hyparameter-tuning
''' 
Thus, these CNNs were considered and used for this project (as shown in some folders in this repo). 

##Getting Started
For training these CNN models, you can download the dataset from Kaggle by going to this [link](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data). In order to train the models, a very good GPU (such as P100) will be needed for processing and training entire dataset (25000+ images) over 100 epochs. The Shared Computing Cluster (aka SCC) from Boston University was used for this. When using remote GPUs, please upload the files to the server and specify the filepath for where the dataset is, as well as where the results should go. Each model makes use of different scripts and libraries, which is seen below.

###ResNet
-Insert description-

###Mask-RCNN
The Mask-RCNN model segments certain regions of an image via "masking". These regions would be fed into the CNN for feature extraction and classification. The Mask-RCNN library will be used, and it is included in this github repo for convenience. The model script that Mask-RCNN has is edited for the purpose of this project, so no need to clone from matterport (clone command shown in appendix however). 

For SCC, multiple processors are needed for training the model. 16 GPUs seemed to be the right amount of processors needed to train the model without running into any interupts or errors. To request this, first type into the terminal:
'''
qrsh -P ece601 -l gpus=0.0625  -l gpu_c=6 -pe omp 16
'''
After this, two modules (python THEN tensorflow) is loaded for use:
'''
module load python/3.6.0
module load tensorflow
'''
Finally, the main script (MaskMain.py) is run (make sure you are in the correct directory):
'''
python MaskMain.py
'''
The entire training process takes appromixately 2 hours. Please specify the DATA_DIR filepath with the dataset lies, and multiple h5 files (representing training weights) will be outputted into the ROOT_DIR filepath. The script will also create a new "testing" model, load the latest weights to the model, and predict pneumonia bounding boxes for the 1000 test images (results in submission.csv).

###ChexNet
-Insert description-

##Applications
Two applications were made for this project, a console app and a web app.

###Console App
The console app script will need h5 files (weights) for each model, as well as JSON files for saved ResNet and ChexNet models (The MaskRCNN model will be created within the script). A folder with the 1000 test images will also be needed for running this app. To start the app, type into the console:
'''
python console_application.py
'''
This will need to a requested input for selecting an option (predict a test image using one of the three models, or look at the loss statistics of each model). If an option of predicting a test image is selected, the user will need the input the specific patient ID. The script will then look for the dicom file with the ID, and predict whether the patient has pneumonia or not.

###Web App
The web app is currently in process, but is using Flask and Google Cloud Platform (GCP). A tutorial is followed [here](https://medium.freecodecamp.org/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492) and the progress is shown [here](https://united-aviary-223117.appspot.com/).

 


