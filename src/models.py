#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The models used for music generation.
"""


def vae_sampling(args):
    from keras import backend as K
    z_mean, z_log_sigma_sq, vae_b1 = args
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.0, stddev=vae_b1)
    return z_mean + K.exp(z_log_sigma_sq * 0.5) * epsilon


def create_keras_autoencoder_model(input_shape, latent_space_size, dropout_rate, max_windows, batchnorm_momentum, use_vae=False, vae_b1=0.02):
    """
    Create larger autoencoder with the options of making it variational and embedding.
    :param input_shape:
    :param latent_space_size:
    :param dropout_rate:
    :param max_windows:
    :param batchnorm_momentum:
    :param use_vae:
    :param vae_b1:
    :return:
    """
    from keras import backend as K
    from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, TimeDistributed, Lambda
    from keras.layers.normalization import BatchNormalization
    from keras.models import Model

    x_in = Input(shape=input_shape)
    print((None,) + input_shape)

    x = Reshape((input_shape[0], -1))(x_in)
    print(K.int_shape(x))

    x = TimeDistributed(Dense(2000, activation='relu'))(x)
    print(K.int_shape(x))

    x = TimeDistributed(Dense(200, activation='relu'))(x)
    print(K.int_shape(x))

    x = Flatten()(x)
    print(K.int_shape(x))

    x = Dense(1600, activation='relu')(x)
    print(K.int_shape(x))

    if use_vae:
        z_mean = Dense(latent_space_size)(x)
        z_log_sigma_sq = Dense(latent_space_size)(x)
        x = Lambda(vae_sampling, output_shape=(latent_space_size,), name='encoder')([z_mean, z_log_sigma_sq, vae_b1])
    else:
        x = Dense(latent_space_size)(x)
        x = BatchNormalization(momentum=batchnorm_momentum, name='encoder')(x)
    print(K.int_shape(x))

    # LATENT SPACE

    x = Dense(1600, name='decoder')(x)
    x = BatchNormalization(momentum=batchnorm_momentum)(x)
    x = Activation('relu')(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    print(K.int_shape(x))

    x = Dense(max_windows * 200)(x)
    print(K.int_shape(x))
    x = Reshape((max_windows, 200))(x)
    x = TimeDistributed(BatchNormalization(momentum=batchnorm_momentum))(x)
    x = Activation('relu')(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    print(K.int_shape(x))

    x = TimeDistributed(Dense(2000))(x)
    x = TimeDistributed(BatchNormalization(momentum=batchnorm_momentum))(x)
    x = Activation('relu')(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    print(K.int_shape(x))

    x = TimeDistributed(Dense(input_shape[1] * input_shape[2], activation='sigmoid'))(x)
    print(K.int_shape(x))
    x = Reshape((input_shape[0], input_shape[1], input_shape[2]))(x)
    print(K.int_shape(x))

    model = Model(x_in, x)

    print(model.summary())

    return model


def create_pytorch_autoencoder_model(input_shape, latent_space_size, dropout_rate, max_windows, batchnorm_momentum):

    import torch.nn as nn
    # Create Encoder and Decoder that subclasses nn.Module

    class TimeDistributed(nn.Module):
        """
        A module to mimic the time distributed wrapper of keras.
        It basically just applies the inserted module to a number of timeseries.
        """
        def __init__(self, module):
            super(TimeDistributed, self).__init__()
            self.module = module

        def forward(self, x):
            if len(x.size()) <= 2:
                return self.module(x)
            t, n = x.size(0), x.size(1)
            # merge batch and seq dimensions
            x_reshape = x.contiguous().view(t * n, x.size(2))
            y = self.module(x_reshape)
            # We have to reshape Y
            y = y.contiguous().view(t, n, y.size()[1])
            return y

    class Encoder(nn.Module):
        """Encoder"""

        LAYER_DIMS = [96 * 96, 2000, 200, 1600]

        def __init__(self):
            super(Encoder, self).__init__()

            self.timedist_relu1 = nn.Sequential(
                TimeDistributed(nn.Sequential(
                    nn.Linear(self.LAYER_DIMS[0], self.LAYER_DIMS[1]),
                    nn.ReLU()
                ))
            )

            self.timedist_relu2 = nn.Sequential(
                TimeDistributed(nn.Sequential(
                    nn.Linear(self.LAYER_DIMS[1], self.LAYER_DIMS[2]),
                    nn.ReLU(),
                )
                )
            )

            self.layer3 = nn.Sequential(
                nn.Linear(self.LAYER_DIMS[2] * max_windows, self.LAYER_DIMS[3]),
                nn.ReLU(),

                nn.Linear(self.LAYER_DIMS[3], latent_space_size),
                nn.BatchNorm1d(latent_space_size, momentum=batchnorm_momentum)
            )

        def forward(self, x):
            # print('input shape', x.shape)

            input = x.view(x.shape[0], input_shape[0], -1)

            # print('encoder flatten1 shape', input.shape)

            out = self.timedist_relu1(input)

            # print('encoder timedist relu1 shape', out.shape)

            out = self.timedist_relu2(out)

            # print('encoder timedist relu2 shape', out.shape)

            out = out.view(out.size(0), -1)

            # print('encoder flatten2 shape', out.shape)

            out = self.layer3(out)

            # print('encoder `linear relu batchnorm shape', out.shape)

            return out

    class Decoder(nn.Module):
        """Decoder"""

        LAYER_DIMS = [1600, 200, 2000]

        def __init__(self):
            super(Decoder, self).__init__()

            self.layer1 = nn.Sequential(
                nn.Linear(latent_space_size, self.LAYER_DIMS[0]),
                nn.BatchNorm1d(self.LAYER_DIMS[0], momentum=batchnorm_momentum),
                nn.ReLU()
            )

            self.dropout1 = nn.Sequential(
                nn.Dropout(dropout_rate)
            )

            self.layer2 = nn.Sequential(
                nn.Linear(self.LAYER_DIMS[0], max_windows * self.LAYER_DIMS[1])
            )

            self.timedist_batchnorm1 = nn.Sequential(
                TimeDistributed(nn.Sequential(
                    nn.BatchNorm1d(self.LAYER_DIMS[1], momentum=batchnorm_momentum)
                ))
            )

            self.layer4 = nn.Sequential(
                nn.ReLU()
            )

            self.dropout2 = nn.Sequential(
                nn.Dropout(dropout_rate)
            )

            self.timedist_linear1 = nn.Sequential(
                TimeDistributed(nn.Sequential(
                    nn.Linear(self.LAYER_DIMS[1], self.LAYER_DIMS[2])
                )
                )
            )

            self.timedist_batchnorm2 = nn.Sequential(
                TimeDistributed(nn.Sequential(
                    nn.BatchNorm1d(self.LAYER_DIMS[2], momentum=batchnorm_momentum)
                ))
            )

            self.relu1 = nn.Sequential(
                nn.ReLU()
            )

            self.dropout3 = nn.Sequential(
                nn.Dropout(dropout_rate)
            )

            self.timedist_sigmoid1 = nn.Sequential(
                TimeDistributed(
                    nn.Sequential(
                        nn.Linear(self.LAYER_DIMS[2], input_shape[1] * input_shape[2]),
                        nn.Sigmoid()
                    )
                )
            )

        def forward(self, x):

            # print('latent space shape', x.shape)

            out = self.layer1(x)

            # print('linear batchnorm relu shape', out.shape)

            if dropout_rate > 0:
                out = self.dropout1(out)

            # print('dropout shape', out.shape)

            out = self.layer2(out)

            # print('linear shape', out.shape)

            out = out.view(out.shape[0], max_windows, self.LAYER_DIMS[1])

            # print('flatten shape', out.shape)

            out = self.timedist_batchnorm1(out)

            # print('batchnorm shape', out.shape)

            out = self.layer4(out)

            # print('relu shape', out.shape)

            if dropout_rate > 0:
                out = self.dropout2(out)

            # print('dropout shape', out.shape)

            out = self.timedist_linear1(out)

            # print('timedist linear shape', out.shape)

            out = self.timedist_batchnorm2(out)

            # print('timedist batchnorm shape', out.shape)

            out = self.relu1(out)

            # print('relu shape', out.shape)

            if dropout_rate > 0:
                out = self.dropout3(out)

            # print('dropout shape', out.shape)

            out = self.timedist_sigmoid1(out)

            # print('sigmoid shape', out.shape)

            out = out.view(out.shape[0], input_shape[0], input_shape[1], input_shape[2])

            # print('output shape', out.shape)

            return out

    encoder = Encoder()

    decoder = Decoder()

    encoder_params = list(p.numel() for p in encoder.parameters())
    decoder_params = list(p.numel() for p in decoder.parameters())
    print("autoencoder with {} parameters (total {})".format(encoder_params + decoder_params, sum(encoder_params + decoder_params)))

    return {'encoder': encoder, 'decoder': decoder}
