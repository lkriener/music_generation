#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural network model abstraction to make it callable.
"""


class AbstractModelLoader:

    def __init__(self):
        pass

    def get_submodel(self, submodel_name):
        pass


class AbstractModel:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class KerasModel(AbstractModel):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        pass

    def predict(self, x):
        pass


