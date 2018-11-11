import matplotlib
from matplotlib import pyplot as plt 
import numpy as np

def plot_stats(model_number=0):
	epochs = np.array([range(100)])
	res_val_loss = np.array([ ])
	res_training_loss = np.array([ ])
	mask_val_loss = np.array([ ])
	mask_training_loss = np.array([ ])
	chex_val_loss = np.array([ ])
	chex_training_loss = np.array([ ])
	epochs = np.reshape(epochs, (100,))
	plt.figure()
	if model_number == 1:
		plt.subplot(1, 2, 1)
		plt.plot(epochs, res_training_loss, label="ResNet Training Loss")
		plt.subplot(1, 2, 2)
		plt.plot(epochs, res_val_loss, label="ResNet Validation Loss")
	if model_number == 2:
		plt.subplot(1, 2, 1)
		plt.plot(epochs, mask_training_loss, label="Mask-RCNN Training Loss")
		plt.subplot(1, 2, 2)
		plt.plot(epochs, mask_val_loss, label="Mask-RCNN Validation Loss")
	if model_number == 3:
		plt.subplot(1, 2, 1)
		plt.plot(epochs, chex_training_loss, label="ChexNet Training Loss")
		plt.subplot(1, 2, 2)
		plt.plot(epochs, chex_val_loss, label="ChexNet Validation Loss")
	plt.show()