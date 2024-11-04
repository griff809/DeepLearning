# Hints:
#
# Load data into appropriate input/output formats.
# For example, for problem 1:
#   * inputs should be a collection of images of size [n_samples x height x width x 3],
#   * outputs should be of size [n_samples x height x width x 1].
#
# Load images using the `np.load` function. You can combine tensors using `np.concat`
# to form a single input or output dataset tensor. Look at previous labs for inspiration
# on what datasets look like.

#
# The amount of training data is very low, so usage of data
# augmentation techniques is strongly advised.(eg. see tf.keras.layers.RandomRotation)
#
# Classification Task hint: you may also think of using pre-trained networks
# with fine-tuning.
# Keep the model size under the restriction of 5Million parameters, you should
# be able to hit the target at much less model size. (Lower the better:))
#
# Segmentation Task hint: You might want to use transpose convolution layers to go up in tensor size.
# Popular segmentation architectures include U-Net (with skip-connections) or DnCNN.
# Keep the model size under the restriction of 3 Million parameters, you should
# be able to hit the target at very much lower model size. (Lower the better:))

#define the good stuff

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers import Conv2DTranspose, Concatenate, UpSampling2D, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
import os

import tensorflow.keras as keras
from tensorflow.keras import datasets
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import model_from_json

from IPython.display import clear_output

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.ticker import MaxNLocator

import pickle
import sklearn as skl

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score

# Define paths to data
data_path = '/home/tcd/data/medicalimaging/dataset'

# Load data for classification
X_benign = np.load(os.path.join(data_path, 'benign/input.npy'))
y_benign = np.zeros((X_benign.shape[0], 3))  # One-hot encoding: [1, 0, 0]
y_benign[:, 0] = 1

X_malignant = np.load(os.path.join(data_path, 'malignant/input.npy'))
y_malignant = np.zeros((X_malignant.shape[0], 3))  # One-hot encoding: [0, 1, 0]
y_malignant[:, 1] = 1

X_normal = np.load(os.path.join(data_path, 'normal/input.npy'))
y_normal = np.zeros((X_normal.shape[0], 3))  # One-hot encoding: [0, 0, 1]
y_normal[:, 2] = 1

# Concatenate all data
X_classification = np.concatenate([X_benign, X_malignant, X_normal])
y_classification = np.concatenate([y_benign, y_malignant, y_normal])

# Load data for segmentation
y_seg_benign = np.load(os.path.join(data_path, 'benign/target.npy'))
y_seg_malignant = np.load(os.path.join(data_path, 'malignant/target.npy'))
y_seg_normal = np.load(os.path.join(data_path, 'normal/target.npy'))

# Concatenate segmentation maps
X_segmentation = np.concatenate([X_benign, X_malignant, X_normal])
y_segmentation = np.concatenate([y_seg_benign, y_seg_malignant, y_seg_normal])

#data augmentation and preprocessing

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data augmentation for classification
datagen_classification = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Define data augmentation for segmentation
datagen_segmentation = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.15
)


#create plot stuff based on previous labs

class PlotLossAccuracy(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.i = 0
        self.x = []
        self.acc = []
        self.losses = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        # Update training history
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

        self.i += 1

        # Clear the output and plot updated data
        clear_output(wait=True)
        plt.figure(figsize=(16, 6))

        # Plot the loss
        plt.subplot(1, 2, 1)
        plt.plot(self.x, self.losses, label="Train Loss")
        plt.plot(self.x, self.val_losses, label="Validation Loss")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Loss')
        plt.legend()

        # Plot the accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.x, self.acc, label="Training Accuracy")
        plt.plot(self.x, self.val_acc, label="Validation Accuracy")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()







#model 1

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2

def build_classification_model():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Freeze the base model layers initially

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        
        
        Dense(3, activation="softmax")
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

classification_model = build_classification_model()

# Check the parameter count to ensure it is within limits
if (classification_model.count_params() < 5000000):
    save_model_to_disk(classification_model, "classification_model")
else:
    print("Your model is unnecessarily complex, scale down!")


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Create data generators for training and validation
train_gen = datagen_classification.flow(X_train, y_train, batch_size=16)
val_gen = datagen_classification.flow(X_val, y_val, batch_size=16)

# Train the model with the modified structure
plot_loss_accuracy = PlotLossAccuracy()

classification_model.fit(
    train_gen,
    epochs=40,
    validation_data=val_gen,
    callbacks=[plot_loss_accuracy]
)




#model 2

def build_segmentation_model():
    inputs = Input((128, 128, 3))
    
    # Down-sampling path
    c1 = Conv2D(16, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001))(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(16, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001))(c1)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.3)(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001))(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001))(c2)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.3)(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001))(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001))(c3)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.3)(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    c4 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001))(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001))(c4)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.3)(c4)
    
    # Up-sampling path with skip connections
    u5 = UpSampling2D((2, 2))(c4)
    u5 = Concatenate()([u5, c3])
    c5 = Conv2D(64, (3, 3), activation="relu", padding="same")(u5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(64, (3, 3), activation="relu", padding="same")(c5)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c2])
    c6 = Conv2D(32, (3, 3), activation="relu", padding="same")(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(32, (3, 3), activation="relu", padding="same")(c6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.3)(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c1])
    c7 = Conv2D(16, (3, 3), activation="relu", padding="same")(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(16, (3, 3), activation="relu", padding="same")(c7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.3)(c7)
    
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c7)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model



segmentation_model = build_segmentation_model()

if (segmentation_model.count_params()  < 3000000) :
  save_model_to_disk(segmentation_model, "segmentation_model")
else:
  print("Your model is unecessarily complex, scale down!")

X_seg_train, X_seg_val, y_seg_train, y_seg_val = train_test_split(X_segmentation, y_segmentation, test_size=0.2, random_state=42)

# Create data generators for training and validation
train_gen_segmentation = datagen_segmentation.flow(X_seg_train, y_seg_train, batch_size=32)
val_gen_segmentation = datagen_segmentation.flow(X_seg_val, y_seg_val, batch_size=32)


# Callback to plot loss and accuracy
plot_loss_accuracy_segmentation = PlotLossAccuracy()

# Train the segmentation model using the separate training and validation generators, with the callback
segmentation_model.fit(
    train_gen_segmentation,
    epochs=20,
    validation_data=val_gen_segmentation,
    callbacks=[plot_loss_accuracy_segmentation]
)