#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural network model and loader implementation for pytorch.
"""

from src.composer.model_abstraction import AbstractModelLoader, AbstractModel


class PyTorchModelLoader(AbstractModelLoader):

    def __init__(self, folder_name):
        super().__init__()

    def get_submodel(self, submodel_name):
        pass


class PyTorchModel(AbstractModel):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        pass
