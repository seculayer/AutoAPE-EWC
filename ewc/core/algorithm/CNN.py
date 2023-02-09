from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense, MaxPooling2D, Conv2D
import numpy as np


class CNN(object):

    def __init__(self, input_shape, lr):

        self.model = Sequential()
        self.input_shape = input_shape
        self.lr = lr

    def create_model(self):
        self.model.add(Conv2D(256, 5, input_shape=self.input_shape, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(128, 5, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, 5, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))


    def compile(self, optimizer, loss_fn, metrics):

        self.model.compile(loss=loss_fn, optimizer = optimizer, metrics=metrics)

    def summary(self):

        self.model.summary()


    def predict(self, data):
        pred = self.model.predict(data)
        return np.argmax(pred, 1)

    def model_save(self, name):
        self.model.save('ewc/resource/' + name + '.h5')
