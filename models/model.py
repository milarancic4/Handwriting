# import the necessary packages
from keras.layers import GlobalAvgPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

def cnn1(height, width, depth, classes):
    inputShape = (height, width, depth)
    model = Sequential([
        Conv2D(filters=64, kernel_size=(5, 5), input_shape=inputShape, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(64, activation='relu'),
        Dense(classes, activation='softmax')
    ])
    return model

def cnn2(height, width, depth, classes):
    inputShape = (height, width, depth)
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), input_shape=inputShape, activation='relu'),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        GlobalAvgPool2D(),
        Dense(classes, activation='softmax')
    ])
    return model
def cnn3(height, width, depth, classes):
    inputShape = (height, width, depth)
    model = Sequential([
        Conv2D(filters=32, kernel_size=(5, 5), input_shape=inputShape, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),

        Dropout(0.25),
        Dense(classes, activation='softmax')
    ])
    return model
def cnn4(height, width, depth, classes):
    inputShape = (height, width, depth)
    model = Sequential([
        Conv2D(filters=64, kernel_size=(5, 5), input_shape=inputShape, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.25),
        Dense(classes, activation='softmax')
    ])
    return model
def cnn5(height, width, depth, classes):
    inputShape = (height, width, depth)
    model = Sequential([
        Conv2D(filters=32, kernel_size=(5, 5), input_shape=inputShape, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),

        Dropout(0.25),
        Dense(classes, activation='softmax')
    ])
    return model