# -*- coding: utf-8 -*-
# Author : JunHyuck Kim
# e-mail : junhyuck.kim@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

from pycmmn.Singleton import Singleton
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

class Constants(object, metaclass=Singleton):
    CONSTANT_VARIABLE: str = "Constant Variable"
    MODEL_RESOURCE_PATH = "ewc/resource"
    LEARNING_RATE = 0.001
    OPTIMIZER = Adam(learning_rate=LEARNING_RATE)
    LOSS_FN=CategoricalCrossentropy(from_logits=True)

