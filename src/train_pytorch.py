#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train an autoencoder model to learn to encode songs.
"""

import random

import numpy as np
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR

import src.midi_utils as midi_utils
import src.plot_utils as plot_utils
import src.models as models

import argparse

#  Load PyTorch
print("Loading pytorch...")
import os
import torch

print("Torch version: " + torch.__version__)

import torch.nn.functional as F

cuda_available = torch.cuda.is_available()
print('with cuda support:', cuda_available)  # to know if it is available

BASE_FOLDER = './'
EPOCHS_QTY = 2000
EPOCHS_TO_SAVE = []  # [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450]
LEARNING_RATE = 0.001  # learning rate
CONTINUE_TRAIN = False
GENERATE_ONLY = False

WRITE_HISTORY = True
NUM_RAND_SONGS = 10

# network params
DROPOUT_RATE = 0.1
BATCHNORM_MOMENTUM = 1-0.9  # weighted normalization with the past (pytorch batchnorm is 1-keras_batchnorm momentum: https://github.com/pytorch/examples/issues/289)
USE_VAE = False
VAE_B1 = 0.02
VAE_B2 = 0.1

BATCH_SIZE = 350
MAX_WINDOWS = 16  # the maximal number of measures a song can have
LATENT_SPACE_SIZE = 120
NUM_OFFSETS = 1

# Fix the random seed so that training comparisons are easier to make
np.random.seed(0)
random.seed(0)


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


def save_training_config(num_songs, optimizer, learning_rate):
    """
    Save configuration of training.
    :param num_songs:
    :param optimizer:
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
        file_out.write('optimizer:           ' + type(optimizer).__name__ + '\n')


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
        random_latent_x = torch.tensor(random_latent_x, dtype=torch.float)

        if cuda_available:
            random_latent_x = random_latent_x.cuda()
        y_song = decoder(random_latent_x).detach().cpu().numpy()[0]
        midi_utils.samples_to_midi(y_song, write_dir + 'random_vectors' + str(i) + '.mid', 32)


def calculate_and_store_pca_statistics(encoder, y_orig, write_dir):
    """
    Calculate means, stddevs, covariance singular values (pca values), covariance singular vectors (pca vectors)
    to more efficiently navigate/find configurations in the latent space.
    :param encoder:
    :param y_orig:
    :param write_dir:
    :return:
    """
    latent_x = np.squeeze(encoder(y_orig).detach().cpu().numpy())

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


def train(samples_path='data/interim/samples.npy', lengths_path='data/interim/lengths.npy', epochs_qty=EPOCHS_QTY, learning_rate=LEARNING_RATE):
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

    y_test_song = torch.tensor(y_test_song, dtype=torch.float)

    if cuda_available:
        y_test_song = y_test_song.cuda()

    #  create model
    if CONTINUE_TRAIN or GENERATE_ONLY:
        print("Loading model...")
        model = models.create_pytorch_autoencoder_model(input_shape=y_shape[1:],
                                                        latent_space_size=LATENT_SPACE_SIZE,
                                                        dropout_rate=DROPOUT_RATE,
                                                        max_windows=MAX_WINDOWS,
                                                        batchnorm_momentum=BATCHNORM_MOMENTUM)
        # saving models (https://pytorch.org/tutorials/beginner/saving_loading_models.html)
        model['encoder'].load_state_dict(torch.load(BASE_FOLDER + 'results/history/encoder.pkl'))
        model['decoder'].load_state_dict(torch.load(BASE_FOLDER + 'results/history/decoder.pkl'))
    else:
        print("Building model...")

        model = models.create_pytorch_autoencoder_model(input_shape=y_shape[1:],
                                                        latent_space_size=LATENT_SPACE_SIZE,
                                                        dropout_rate=DROPOUT_RATE,
                                                        max_windows=MAX_WINDOWS,
                                                        batchnorm_momentum=BATCHNORM_MOMENTUM)

    #  train
    print("Referencing sub-models...")
    encoder = model['encoder']
    decoder = model['decoder']

    if cuda_available:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    parameters = list(encoder.parameters()) + list(decoder.parameters())
    criterion = F.binary_cross_entropy
    optimizer = torch.optim.RMSprop(parameters, lr=learning_rate, alpha=0.9, eps=1e-07)  # same params as keras
    # optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    # optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)

    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    random_vectors = np.random.normal(0.0, 1.0, (NUM_RAND_SONGS, LATENT_SPACE_SIZE))
    np.save(BASE_FOLDER + 'data/interim/random_vectors.npy', random_vectors)

    if GENERATE_ONLY:
        print("Generating songs...")
        generate_normalized_random_songs(y_orig, encoder, decoder, random_vectors, BASE_FOLDER + 'results/')
        for save_epoch in range(20):
            x_test_song = x_train[save_epoch:save_epoch + 1]
            x_test_song = torch.tensor(x_test_song, dtype=torch.float)

            if cuda_available:
                x_test_song = x_test_song.cuda()
            y_song = decoder(encoder(x_test_song))[0]
            midi_utils.samples_to_midi(y_song, BASE_FOLDER + 'results/gt' + str(save_epoch) + '.mid')
        exit(0)

    save_training_config(songs_qty, optimizer, learning_rate)
    print("Training model...")
    train_loss = []
    offset = 0

    for epoch in range(epochs_qty):
        encoder.train()
        decoder.train()

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

        y_train_pt = torch.tensor(y_train, dtype=torch.float)
        y_orig_pt = torch.tensor(y_orig, dtype=torch.float)

        if cuda_available:
            y_train_pt = y_train_pt.cuda()
            y_orig_pt = y_orig_pt.cuda()

        optimizer.zero_grad()
        output = encoder(y_train_pt)
        output = decoder(output)

        loss = criterion(output, y_train_pt)

        loss.backward()
        optimizer.step()

        # store last loss
        loss = loss.data.cpu()
        train_loss.append(loss)
        print("Train loss: " + str(train_loss[-1].numpy()))

        scheduler.step(loss)  # change the learning rate as soon as the loss plateaus

        if WRITE_HISTORY:
            plot_losses(train_loss, BASE_FOLDER + 'results/history/losses.png', True)
        else:
            plot_losses(train_loss, BASE_FOLDER + 'results/losses.png', True)

        # save model periodically
        save_epoch = epoch + 1
        if save_epoch in EPOCHS_TO_SAVE or (save_epoch % 100 == 0) or save_epoch == epochs_qty:
            write_dir = ''
            if WRITE_HISTORY:
                # Create folder to save models into
                write_dir += BASE_FOLDER + 'results/history/e' + str(save_epoch)
                if not os.path.exists(write_dir):
                    os.makedirs(write_dir)
                write_dir += '/'
                torch.save(encoder.state_dict(), BASE_FOLDER + 'results/history/encoder.pkl')
                torch.save(decoder.state_dict(), BASE_FOLDER + 'results/history/decoder.pkl')
            else:
                torch.save(encoder.state_dict(), BASE_FOLDER + 'results/encoder.pkl')
                torch.save(decoder.state_dict(), BASE_FOLDER + 'results/decoder.pkl')

            print("...Saved.")

            encoder.eval()
            decoder.eval()
            y_song = decoder(encoder(y_test_song))[0]

            plot_utils.plot_samples(write_dir + 'test', y_song.detach().cpu().numpy())
            midi_utils.samples_to_midi(y_song, write_dir + 'test.mid')

            generate_normalized_random_songs(y_orig_pt, encoder, decoder, random_vectors, write_dir)

    print("...Done.")


if __name__ == "__main__":
    # configure parser and parse arguments
    parser = argparse.ArgumentParser(description='Train to reconstruct midi in autoencoder.')
    parser.add_argument('--samples_path', default='data/interim/samples.npy', type=str, help='Path to samples numpy array.')
    parser.add_argument('--lengths_path', default='data/interim/lengths.npy', type=str, help='Path to sample lengths numpy array.')
    parser.add_argument('--epochs_qty', default=EPOCHS_QTY, type=int, help='The number of epochs to be trained.')
    parser.add_argument('--learning_rate', default=LEARNING_RATE, type=float, help='The learning rate to train the model.')

    BASE_FOLDER = '../'

    args = parser.parse_args()
    epochs_qty = args.epochs_qty
    learning_rate = args.learning_rate
    samples_path = args.samples_path
    lengths_path = args.lengths_path
    train(samples_path, lengths_path, epochs_qty, learning_rate)
