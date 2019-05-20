#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural network model and loader implementation for pytorch.
"""

from src.model_loading.model_abstraction import AbstractModelLoader, AbstractModel
import src.models as models
import torch
import os
import numpy as np

LATENT_SPACE_SIZE = 120
DROPOUT_RATE = 0.1
MAX_WINDOWS = 16  # the maximal number of measures a song can have
BATCHNORM_MOMENTUM = 0.9  # weighted normalization with the past


class PyTorchModelLoader(AbstractModelLoader):

    def __init__(self, folder_name, input_shape):
        super().__init__()
        self.model = models.create_pytorch_double_autoencoder_model(input_shape=input_shape,
                                                                    latent_space_size=LATENT_SPACE_SIZE,
                                                                    dropout_rate=DROPOUT_RATE,
                                                                    max_windows=MAX_WINDOWS,
                                                                    batchnorm_momentum=BATCHNORM_MOMENTUM)
        # saving models (https://pytorch.org/tutorials/beginner/saving_loading_models.html)
        self.model['encoder'].load_state_dict(torch.load(os.path.join(folder_name, 'encoder.pkl'), map_location='cpu'))
        self.model['decoder'].load_state_dict(torch.load(os.path.join(folder_name, 'decoder.pkl'), map_location='cpu'))

        self.model['encoder'] = PyTorchModel(self.model['encoder'])
        self.model['decoder'] = PyTorchModel(self.model['decoder'])

    def get_submodel(self, submodel_name):
        return self.model[submodel_name]


class PyTorchModel(AbstractModel):
    def __init__(self, torch_model):
        super().__init__()
        self.torch_model = torch_model
        self.torch_model.eval()

    def __call__(self, *args, **kwargs):
        x = args[0][0]
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        return [self.torch_model(x).detach().cpu().numpy(), 0]

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        return self.torch_model(x).detach().cpu().numpy()

