import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from sklearn.utils import shuffle
import os



def display_title_bar():
    # Clears the terminal screen, and displays a title bar.
    os.system('clear')

    print("\t**********************************************")
    print("\t***  Predictor - Detect Lung Opacities!  ***")
    print("\t**********************************************")

def get_user_choice():
    # Let users know what they can do.
    print("\n[1] ResNet")
    print("[2] MaskRCNN")
    print("[3] ChexNet")
    print("[q] Quit.")

    return input("Which model would you like to use? ")


choice = ''
display_title_bar()
while choice != 'q':

    choice = get_user_choice()

    # Respond to the user's choice.
    display_title_bar()
    if choice == '1':
        # Load ResNet Model
        pass
    elif choice == '2':
        # Load MaskRCNN
        pass
    elif choice == '3':
        # Load ChexNet
        pass
    elif choice == 'q':
        print("\nBye.")
    else:
        print("\nI didn't understand that choice.\n")
