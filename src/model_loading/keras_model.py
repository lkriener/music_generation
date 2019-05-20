#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural network model and loader implementation for keras.
"""

import os
import keras
from keras.models import Model, load_model
from keras import backend as K

print("Keras version: " + keras.__version__)

K.set_image_data_format('channels_first')

from src.model_loading.model_abstraction import AbstractModelLoader, AbstractModel


class KerasModelLoader(AbstractModelLoader):

    def __init__(self, folder_name):
        super().__init__()
        self.model = load_model(os.path.join(folder_name, 'model.h5'))
        self.sub_models = {
            'encoder': KerasModel(Model(inputs=self.model.input, outputs=self.model.get_layer('encoder').output)),
            'decoder': KerasModel(K.function([self.model.get_layer('decoder').input, K.learning_phase()],
                                             [self.model.layers[-1].output]))
        }

    def get_submodel(self, submodel_name):
        return self.sub_models[submodel_name]


class KerasModel(AbstractModel):
    def __init__(self, keras_model):
        super().__init__()
        self.keras_model = keras_model

    def __call__(self, *args, **kwargs):
        return self.keras_model(*args)

    def predict(self, x):
        return self.keras_model.predict(x)
