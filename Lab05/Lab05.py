# Import the necessary modules

import tensorflow as tf

import keras
from keras import datasets
from keras.layers import Dense, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import PReLU, LeakyReLU, Conv2D, MaxPool2D, Lambda
from keras.regularizers import l2

from keras.models import model_from_json

from IPython.display import clear_output

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib.ticker import MaxNLocator

import pickle
import sklearn as skl

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score

import subprocess


import os
import subprocess
import pickle
import tensorflow as tf
from keras import datasets
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from matplotlib.ticker import MaxNLocator

# Define some useful functions
class PlotLossAccuracy(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.acc = []
        self.losses = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(int(self.i))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

        self.i += 1

        plt.figure(figsize=(16, 6))
        plt.subplot(121)
        plt.plot(self.x, self.losses, label="train loss")
        plt.plot(self.x, self.val_losses, label="validation loss")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title('Model Loss')
        plt.legend()
        plt.subplot(122)
        plt.plot(self.x, self.acc, label="training accuracy")
        plt.plot(self.x, self.val_acc, label="validation accuracy")
        plt.legend()
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.title('Model Accuracy')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()

def save_model_to_disk(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to model.json and weights to model.h5")

# File path to the dataset
dataset_path = "/home/tcd/data/cifar10-dataset.pkl"

# Check if the dataset already exists
if not os.path.exists(dataset_path):
    print("Dataset not found, downloading the dataset...")
    subprocess.run([
        "curl", "--create-dirs", 
        "-o", dataset_path, 
        "https://tcddeeplearning.blob.core.windows.net/deeplearning202324/cifar10-dataset.pkl"
    ])
else:
    print("Dataset already exists. Skipping download.")

# Load the dataset
try:
    with open(dataset_path, 'rb') as pkl_file:
        dataset = pickle.load(pkl_file)
    print("Dataset loaded successfully.")
except pickle.UnpicklingError:
    print("Error loading the dataset. The file might be corrupt.")

# Prepare the dataset
X = dataset['X'].astype('float32') / 255
Y = dataset['Y'].astype('float32')
Y = tf.keras.utils.to_categorical(Y)

# Visualize some images
plt.figure(figsize=(12, 12))
for i in range(9):
    pic = X[i]
    classid = Y[i].argmax(-1)
    classname = dataset['labels'][classid]
    plt.subplot(3, 3, i + 1)
    plt.imshow(pic)
    plt.title(f'label: {classname}')
plt.show()




