# EC601_A2_Pneumonia_Detection

The goal of this project is to detect pneumonia in Chest X Rays (CXRs) by detecting opacities in the lungs.

We've currently tried out 3 different CNNs: ResNet, ChexNet and Mask RCNN. Each of the notebooks that we used 
for testing are in this repo, each in a separate folder. 

Each notebook was originally from Kaggle:
  - Mask RCNN: https://www.kaggle.com/drt2290078/mask-rcnn-sample-starter-code
  - ChexNet: https://www.kaggle.com/ashishpatel26/chexnet-batch-normalization-hyparameter-tuning
  - ResNet: https://www.kaggle.com/uds5501/cnn-segmentation-resnet-depth-5

Each of these notebooks were modified for our first round of testing to use 10 epochs and a training set of 10000 images and a
validation set of 1000 images, so that they were able to run on the with the GPU provided by the Kaggle notebooks.
