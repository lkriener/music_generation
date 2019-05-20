#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train an autoencoder model to learn to encode songs.
"""

import random

import numpy as np
from matplotlib import pyplot as plt

import src.midi_utils as midi_utils
import src.plot_utils as plot_utils
import src.models as models

import argparse

#  Load Keras
print("Loading keras...")
import os
import keras

print("Keras version: " + keras.__version__)

from keras.models import Model, load_model
from keras.utils import plot_model
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, RMSprop

BASE_FOLDER = './'
EPOCHS_QTY = 2000
EPOCHS_TO_SAVE_PCA_STATS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450]
EPOCHS_TO_SAVE_MODEL = []
LEARNING_RATE = 0.001  # learning rate
CONTINUE_TRAIN = False
GENERATE_ONLY = False

WRITE_HISTORY = True
NUM_RAND_SONGS = 10

# network params
USE_DOUBLE_AUTOENCODER = False
DROPOUT_RATE = 0.1
BATCHNORM_MOMENTUM = 0.9  # weighted normalization with the past
USE_VAE = False
VAE_B1 = 0.02
VAE_B2 = 0.1

BATCH_SIZE = 350
MAX_WINDOWS = 16  # the maximal number of measures a song can have
LATENT_SPACE_SIZE = 120
NUM_OFFSETS = 1

K.set_image_data_format('channels_first')

# Fix the random seed so that training comparisons are easier to make
np.random.seed(0)
random.seed(0)


def vae_loss(x, x_decoded_mean, z_log_sigma_sq, z_mean):
    """
    Variational autoencoder loss function.
    :param x:
    :param x_decoded_mean:
    :param z_log_sigma_sq:
    :param z_mean:
    :return:
    """
    xent_loss = binary_crossentropy(x, x_decoded_mean)
    kl_loss = VAE_B2 * K.mean(1 + z_log_sigma_sq - K.square(z_mean) - K.exp(z_log_sigma_sq), axis=None)
    return xent_loss - kl_loss


def plot_losses(scores, f_name, on_top=True):
    """
    Plot loss.
    :param scores:
    :param f_name:
    :param on_top:
    :return:
    """
    plt.clf()
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.grid(True)
    plt.plot(scores)
    plt.ylim([0.0, 0.009])
    plt.xlabel('Epoch')
    loc = ('upper right' if on_top else 'lower right')
    plt.draw()
    plt.savefig(f_name)


def save_training_config(num_songs, model, learning_rate):
    """
    Save configuration of training.
    :param num_songs:
    :param model:
    :param learning_rate:
    :return:
    """
    with open(BASE_FOLDER + 'results/config.txt', 'w') as file_out:
        file_out.write('LEARNING_RATE:       ' + str(learning_rate) + '\n')
        file_out.write('BATCHNORM_MOMENTUM:  ' + str(BATCHNORM_MOMENTUM) + '\n')
        file_out.write('BATCH_SIZE:          ' + str(BATCH_SIZE) + '\n')
        file_out.write('NUM_OFFSETS:         ' + str(NUM_OFFSETS) + '\n')
        file_out.write('DROPOUT_RATE:        ' + str(DROPOUT_RATE) + '\n')
        file_out.write('num_songs:           ' + str(num_songs) + '\n')
        file_out.write('optimizer:           ' + type(model.optimizer).__name__ + '\n')


def generate_random_songs(decoder, write_dir, random_vectors):
    """
    Generate random songs using random latent vectors.
    :param decoder:
    :param write_dir:
    :param random_vectors:
    :return:
    """
    for i in range(random_vectors.shape[0]):
        random_latent_x = random_vectors[i:i + 1]
        y_song = decoder([random_latent_x, 0])[0]
        midi_utils.samples_to_midi(y_song[0], write_dir + 'random_vectors' + str(i) + '.mid', 32)


def calculate_and_store_pca_statistics(encoder, y_orig, write_dir):
    """
    Calculate means, stddevs, covariance singular values (pca values), covariance singular vectors (pca vectors)
    to more efficiently navigate/find configurations in the latent space.
    :param encoder:
    :param y_orig:
    :param write_dir:
    :return:
    """
    latent_x = np.squeeze(encoder.predict(y_orig))

    latent_mean = np.mean(latent_x, axis=0)
    latent_stds = np.std(latent_x, axis=0)
    latent_cov = np.cov((latent_x - latent_mean).T)
    _, latent_pca_values, latent_pca_vectors = np.linalg.svd(latent_cov)
    latent_pca_values = np.sqrt(latent_pca_values)

    print("Latent Mean values: ", latent_mean[:6])
    print("Latent PCA values: ", latent_pca_values[:6])

    np.save(write_dir + 'latent_means.npy', latent_mean)
    np.save(write_dir + 'latent_stds.npy', latent_stds)
    np.save(write_dir + 'latent_pca_values.npy', latent_pca_values)
    np.save(write_dir + 'latent_pca_vectors.npy', latent_pca_vectors)
    return latent_mean, latent_stds, latent_pca_values, latent_pca_vectors


def generate_normalized_random_songs(y_orig, encoder, decoder, random_vectors, write_dir):
    """
    Generate a number of random songs from some normal latent vector samples.
    :param encoder:
    :param y_orig:
    :param decoder:
    :param write_dir:
    :param random_vectors:
    :return:
    """
    latent_mean, latent_stds, pca_values, pca_vectors = calculate_and_store_pca_statistics(encoder, y_orig, write_dir)

    latent_vectors = latent_mean + np.dot(random_vectors * pca_values, pca_vectors)
    generate_random_songs(decoder, write_dir, latent_vectors)

    title = ''
    if '/' in write_dir:
        title = 'Epoch: ' + write_dir.split('/')[-2][1:]

    plt.clf()
    pca_values[::-1].sort()
    plt.title(title)
    plt.bar(np.arange(pca_values.shape[0]), pca_values, align='center')
    plt.draw()
    plt.savefig(write_dir + 'latent_pca_values.png')

    plt.clf()
    plt.title(title)
    plt.bar(np.arange(pca_values.shape[0]), latent_mean, align='center')
    plt.draw()
    plt.savefig(write_dir + 'latent_means.png')

    plt.clf()
    plt.title(title)
    plt.bar(np.arange(pca_values.shape[0]), latent_stds, align='center')
    plt.draw()
    plt.savefig(write_dir + 'latent_stds.png')


def train(samples_path='data/interim/samples.npy', lengths_path='data/interim/lengths.npy', use_double_autoencoder=USE_DOUBLE_AUTOENCODER, epochs_qty=EPOCHS_QTY, learning_rate=LEARNING_RATE):
    """
    Train model.
    :return:
    """

    # Create folders to save models into
    if not os.path.exists(BASE_FOLDER + 'results'):
        os.makedirs(BASE_FOLDER + 'results')
    if WRITE_HISTORY and not os.path.exists(BASE_FOLDER + 'results/history'):
        os.makedirs(BASE_FOLDER + 'results/history')

    # Load dataset into memory
    print("Loading Data...")
    if not os.path.exists(samples_path) or not os.path.exists(lengths_path):
        print('No input data found, run preprocess_songs.py first.')
        exit(1)

    y_samples = np.load(samples_path)
    y_lengths = np.load(lengths_path)

    samples_qty = y_samples.shape[0]
    songs_qty = y_lengths.shape[0]
    print("Loaded " + str(samples_qty) + " samples from " + str(songs_qty) + " songs.")
    print(np.sum(y_lengths))
    assert (np.sum(y_lengths) == samples_qty)

    print("Preparing song samples, padding songs...")
    x_shape = (songs_qty * NUM_OFFSETS, 1)  # for embedding
    x_orig = np.expand_dims(np.arange(x_shape[0]), axis=-1)

    y_shape = (songs_qty * NUM_OFFSETS, MAX_WINDOWS) + y_samples.shape[1:]  # (songs_qty, max number of windows, window pitch qty, window beats per measure)
    y_orig = np.zeros(y_shape, dtype=y_samples.dtype)  # prepare dataset array

    # fill in measure of songs into input windows for network
    song_start_ix = 0
    song_end_ix = y_lengths[0]
    for song_ix in range(songs_qty):
        for offset in range(NUM_OFFSETS):
            ix = song_ix * NUM_OFFSETS + offset  # calculate the index of the song with its offset
            song_end_ix = song_start_ix + y_lengths[song_ix]  # get song end ix
            for window_ix in range(MAX_WINDOWS):  # get a maximum number of measures from a song
                song_measure_ix = (window_ix + offset) % y_lengths[song_ix]  # chosen measure of song to be placed in window (modulo song length)
                y_orig[ix, window_ix] = y_samples[song_start_ix + song_measure_ix]  # move measure into window
        song_start_ix = song_end_ix  # new song start index is previous song end index
    assert (song_end_ix == samples_qty)
    x_train = np.copy(x_orig)
    y_train = np.copy(y_orig)

    # copy some song from the samples and write it to midi again
    test_ix = 0
    y_test_song = np.copy(y_train[test_ix: test_ix + 1])
    midi_utils.samples_to_midi(y_test_song[0], BASE_FOLDER + 'data/interim/gt.mid')

    #  create model
    if CONTINUE_TRAIN or GENERATE_ONLY:
        print("Loading model...")
        model = load_model(BASE_FOLDER + 'results/history/model.h5')
    else:
        print("Building model...")

        if use_double_autoencoder:
            model = models.create_keras_double_autoencoder_model(input_shape=y_shape[1:],
                                                                 latent_space_size=LATENT_SPACE_SIZE,
                                                                 dropout_rate=DROPOUT_RATE,
                                                                 max_windows=MAX_WINDOWS,
                                                                 batchnorm_momentum=BATCHNORM_MOMENTUM,
                                                                 use_vae=USE_VAE,
                                                                 vae_b1=VAE_B1)
        else:
            model = models.create_keras_autoencoder_model(input_shape=y_shape[1:],
                                                          latent_space_size=LATENT_SPACE_SIZE,
                                                          dropout_rate=DROPOUT_RATE,
                                                          max_windows=MAX_WINDOWS,
                                                          batchnorm_momentum=BATCHNORM_MOMENTUM,
                                                          use_vae=USE_VAE,
                                                          vae_b1=VAE_B1)

        if USE_VAE:
            model.compile(optimizer=Adam(lr=learning_rate), loss=vae_loss)
        else:
            model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')

        # plot model with graphvis if installed
        try:
            plot_model(model, to_file='results/model.png', show_shapes=True)
        except OSError as e:
            print(e)

    #  train
    print("Referencing sub-models...")
    decoder = K.function([model.get_layer('decoder').input, K.learning_phase()], [model.layers[-1].output])
    encoder = Model(inputs=model.input, outputs=model.get_layer('encoder').output)

    random_vectors = np.random.normal(0.0, 1.0, (NUM_RAND_SONGS, LATENT_SPACE_SIZE))
    np.save(BASE_FOLDER + 'data/interim/random_vectors.npy', random_vectors)

    if GENERATE_ONLY:
        print("Generating songs...")
        generate_normalized_random_songs(y_orig, encoder, decoder, random_vectors, BASE_FOLDER + 'results/')
        for save_epoch in range(20):
            x_test_song = x_train[save_epoch:save_epoch + 1]
            y_song = model.predict(x_test_song, batch_size=BATCH_SIZE)[0]
            midi_utils.samples_to_midi(y_song, BASE_FOLDER + 'results/gt' + str(save_epoch) + '.mid')
        exit(0)

    save_training_config(songs_qty, model, learning_rate)
    print("Training model...")
    train_loss = []
    offset = 0

    for epoch in range(epochs_qty):
        print("Training epoch: ", epoch, "of", epochs_qty)
        # produce songs from its samples with a different starting point of the song each time
        song_start_ix = 0
        for song_ix in range(songs_qty):
            song_end_ix = song_start_ix + y_lengths[song_ix]
            for window_ix in range(MAX_WINDOWS):
                song_measure_ix = (window_ix + offset) % y_lengths[song_ix]
                y_train[song_ix, window_ix] = y_samples[song_start_ix + song_measure_ix]
            song_start_ix = song_end_ix
        assert (song_end_ix == samples_qty)
        offset += 1

        history = model.fit(y_train, y_train, batch_size=BATCH_SIZE, epochs=1)  # train model on reconstruction loss

        # store last loss
        loss = history.history["loss"][-1]
        train_loss.append(loss)
        print("Train loss: " + str(train_loss[-1]))

        if WRITE_HISTORY:
            plot_losses(train_loss, BASE_FOLDER + 'results/history/losses.png', True)
        else:
            plot_losses(train_loss, BASE_FOLDER + 'results/losses.png', True)

        # save model periodically either to not lose the last model or for explicitly keeping the state of a certain epoch
        save_epoch = epoch + 1
        store_stats = False
        if save_epoch in EPOCHS_TO_SAVE_MODEL or (save_epoch % 100 == 0) or save_epoch == epochs_qty:
            write_dir = ''
            if WRITE_HISTORY:
                # Create folder to save models into
                write_dir += BASE_FOLDER + 'results/history/e' + str(save_epoch)
                if not os.path.exists(write_dir):
                    os.makedirs(write_dir)
                write_dir += '/'
                model.save(BASE_FOLDER + write_dir + 'model.h5')

                store_stats = True

                print("...Saved model separately to keep current state.")

        if save_epoch in EPOCHS_TO_SAVE_PCA_STATS or (save_epoch % 100 == 0) or save_epoch == epochs_qty or store_stats:
            write_dir = ''
            if WRITE_HISTORY:
                # Create folder to save models into
                write_dir += BASE_FOLDER + 'results/history/e' + str(save_epoch)
                if not os.path.exists(write_dir):
                    os.makedirs(write_dir)
                write_dir += '/'
                model.save(BASE_FOLDER + 'results/history/model.h5')
            else:
                model.save(BASE_FOLDER + 'results/model.h5')

            print("...Saved PCA statistics and last model.")

            y_song = model.predict(y_test_song, batch_size=BATCH_SIZE)[0]

            plot_utils.plot_samples(write_dir + 'test', y_song)
            midi_utils.samples_to_midi(y_song, write_dir + 'test.mid')

            generate_normalized_random_songs(y_orig, encoder, decoder, random_vectors, write_dir)

    print("...Done.")

    return model, train_loss


if __name__ == "__main__":
    # configure parser and parse arguments
    parser = argparse.ArgumentParser(description='Train to reconstruct midi in autoencoder.')
    parser.add_argument('--use_double_autoencoder', default=USE_DOUBLE_AUTOENCODER, action='store_true', help='Double autoencoder or simple autoencoder.')
    parser.add_argument('--samples_path', default='data/interim/samples.npy', type=str, help='Path to samples numpy array.')
    parser.add_argument('--lengths_path', default='data/interim/lengths.npy', type=str, help='Path to sample lengths numpy array.')
    parser.add_argument('--epochs_qty', default=EPOCHS_QTY, type=int, help='The number of epochs to be trained.')
    parser.add_argument('--learning_rate', default=LEARNING_RATE, type=float, help='The learning rate to train the model.')

    BASE_FOLDER = '../'

    args = parser.parse_args()
    samples_path = args.samples_path
    lengths_path = args.lengths_path
    use_double_autoencoder = args.use_double_autoencoder
    epochs_qty = args.epochs_qty
    learning_rate = args.learning_rate
    train(samples_path, lengths_path, use_double_autoencoder, epochs_qty, learning_rate)
