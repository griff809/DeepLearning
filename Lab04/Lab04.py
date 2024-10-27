# the data structure is a tensor, ie. it is a multidimensional array
# each layer instance is callable on a tensor, and returns a tensor

# The model below contains 2 hidden layers with 30 nodes each.
# The activation functions for these 2 layers is the ReLU
# The network ends with a 10 nodes layer with softmax activation
# The first 2 hidden layers transform the original features into
# a new feature vector of size 30.
# The last layer essentially does the classification using multonomial regression
# based on these new features.
from keras.layers import BatchNormalization

inputs = keras.layers.Input(shape=(32, 32, 3))
x = Flatten()(inputs)
x = Dense(1000, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(500, activation='relu')(x)
x = Dropout(0.2)(x)

x = Dense(500, activation='relu')(x)

x = Dense(500, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)

x = Dense(300, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)

x = Dense(500, kernel_regularizer='l2')(x)
x = Dropout(0.1)(x)

x = Dense(500, kernel_regularizer='l2')(x)
x = Dropout(0.1)(x)

keras.regularizers.L2(l2=0.01)

predictions = Dense(10, activation='softmax')(x)

# we create the model
model = keras.models.Model(inputs=inputs, outputs=predictions)
opt = keras.optimizers.legacy.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# setup the optimisation strategy
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.summary()

if (model.count_params() > 5000000):
    raise Exception("Your model is unecessarily complex, scale down!")
