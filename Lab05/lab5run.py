from Lab05.Lab05 import *


# let's split data between validation set and training set

X_train, X_validation, Y_train, Y_validation = skl.model_selection.train_test_split(X, Y, test_size=.1)

# The model below contains 2 hidden layers with 64 nodes each.
# The activation functions for these 2 layers is the ReLU
# The network ends with a 10 nodes layer with softmax activation
# The first 2 hidden layers transform the original features into
# a new feature vector of size 64.
# The last layer essentially does the classification using multonomial regression
# based on these new features.

inputs = keras.layers.Input(shape=(32, 32, 3))


x = inputs  # Don't flatten until after your 2D layers
...
# put here some convolutional layers;
# see keras docs for Conv2D, MaxPool2D, etc.
# Maybe stick in some Dropout as well?

x = Conv2D(500, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D((2, 2))(x)
x = Conv2D(500, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D((2, 2))(x)
x = Conv2D(300, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D((2, 2))(x)

x = Dropout(0.25)(x)




# The last layers of your model should look something like
# this to be well-structured for the final classification:
x = Flatten()(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)



predictions = Dense(10, activation='softmax')(x)  # There are 10 classes in this problem, hence the '10'.



# Create the model.
model = keras.models.Model(inputs=inputs, outputs=predictions)
opt = keras.optimizers.legacy.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Setup the optimisation strategy.
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display a summary.
model.summary()

# Keep things sane.
if (model.count_params() > 4500000):
    raise Exception("Your model is unecessarily complex, scale down!")



# Note that you can evaluate this cell repeatedly to push the training of your model further.
# You might want to reduce the value of 'num_epochs' if each evaluation starts to take too long.

num_epochs = 40

# Create an instance of our callback functions class, to plot our loss function and accuracy with each epoch.
pltCallBack = PlotLossAccuracy()

# Run the training.
model.fit(X_train, Y_train,
          batch_size=1024, epochs=num_epochs,
          validation_data=(X_validation, Y_validation),
          callbacks=[pltCallBack])
