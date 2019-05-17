import src.midi_utils as midi_utils
from src.dataset_utils import TrackDataset, get_dataset_representation_from_tracks

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import glob
import os
import copy
import pygame
import numpy as np
import matplotlib.pyplot as plt


class FeedForward(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, input):
        return self.layers(input)


def load_tracks(dirname, track_nr):
    """
    Load all midi-files from a directory, reduce to single track and convert to numpy
    :param dirname: assuming current working dir is where notebook/script runs
    :param track_nr: number of track that should be kept
    :return: list of numpy-tracks
    """
    tracks = []
    try:
        home_dir
    except NameError:
        home_dir = os.getcwd()

    os.chdir(home_dir + dirname)  # go to a folder relative to home dir
    for midi_file in glob.glob("*.mid"):
        # get a list of all soprano tracks
        ## load midi file
        csv_text = midi_utils.load_to_csv(midi_file)

        ## Split into tracks
        track_dict = midi_utils.split_tracks(csv_text)
        track_nr = '1'

        ## Generating numpy array with notes
        track = midi_utils.midi_track_to_numpy(track_dict[track_nr])
        tracks.append(track)

    print('Loaded {} tracks'.format(len(tracks)))
    return tracks


def generate_dataloaders(tracks, minibatch_size, train_valid_ratio, feature_qty, prediction_qty,
                        interval_range=[-12,12]):
    """
    Produce train- and valid-datasets and dataloaders from numpy_tracks
    :param tracks: 
    :param minibatch_size: 
    :param train_valid_ratio: ratio of data going into training and validation set
    :param feature_qty: how many starter intervals given to the networks 
    :param prediction_qty: how many intervals should network predict
    :param interval_range: available interval range for output (+- one octave reccomendet)
    :return: train- and validation datasets and dataloaders
    """
    assert prediction_qty == 1  #because one-hot encoding in output allows only one prediction
    
    interval_indices = {}
    interval_values = np.arange(interval_range[0], interval_range[1]+1, 1)
    j = 0
    for i in interval_values:
        interval_indices[str(i)] = j
        j += 1

    x, y = get_dataset_representation_from_tracks(tracks, feature_qty=feature_qty, prediction_qty=prediction_qty, intervals=True)
    # drop length of notes and keep pitch
    x = np.stack(x)
    x = x[:,:,0]
    x = x / max(abs(interval_range[0]), interval_range[1])
    y = np.stack(y)
    y = y[:,:,0]
    # we will use cross-entropy loss, so the label must be the interval-index
    y_idx = np.array([int(interval_indices[str(i[0])]) for i in y])

    print("Mean of the dataset: " + str(np.mean(x)))
    print("Number of samples: " + str(len(x)))

    train_indices = np.random.choice(np.arange(0, len(x), 1), int(train_valid_ratio*len(x)), replace=False)
    valid_indices = np.setdiff1d(np.arange(0, len(x), 1), train_indices)
    train_dataset = TrackDataset(x[train_indices], y_idx[train_indices], drop_length=False, intervals=True)  # make training dataset
    valid_dataset = TrackDataset(x[valid_indices], y_idx[valid_indices], drop_length=False, intervals=True)  # make validation dataset

    train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
    validation_loader = DataLoader(valid_dataset, batch_size=minibatch_size, shuffle=True) 
    return train_dataset, valid_dataset, train_loader, validation_loader


def train(model, train_dataloader, val_dataloader, optimizer, n_epochs, 
          loss_function, device=torch.device('cpu'), verbose=True, snapshot=50):
    """
    Trains the given model
    :param model: 
    :param train_dataloader: 
    :param val_dataloader: 
    :param optimizer: 
    :param n_epochs: 
    :param loss_function: 
    :param device: 
    :param verbose: print info during training
    :param snapshot: epoch_nr at which a snapshot-copy of the model is created
    :return: train- and validation losses + model-snapshot
    """
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        losses = []
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            # set gradients to zero
            optimizer.zero_grad()

            loss = loss_function(output, y)
            if verbose > 2:
                print(loss.item())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_losses.append(np.mean(np.array(losses)))
        
        if epoch == snapshot:
            model_snapshot = copy.deepcopy(model)
            
        # Evaluation phase
        model.eval()
        losses = []
        # We don't need gradients
        with torch.no_grad():
            for x, y in val_dataloader:
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                loss = loss_function(output, y)
                losses.append(loss.item())
        val_losses.append(np.mean(np.array(losses)))
        
        if verbose and (epoch % 10 == 0):
            print('Epoch {}/{}: train_loss: {:.4f}, val_loss: {:.4f}'.format(epoch + 1, n_epochs, train_losses[-1], val_losses[-1]))
    return train_losses, val_losses, model_snapshot


def save_results(result_dir, result_name, untrained_model, snapshot_model, 
                 forward_model, train_losses, val_losses, snapshot_epoch):
    """
    Saves the resulting models and the training performance plot
    :param result_dir: Directory to which stuff should be saved
    :param result_name: Prefix of the filenames
    :param untrained_model: 
    :param snapshot_model: 
    :param forward_model: 
    :param train_losses: 
    :param val_losses: 
    :param snapshot_epoch:
    :return:
    """
    savepath = result_dir + result_name + 'untrained.pth'
    torch.save(untrained_model, savepath)
    savepath = result_dir + result_name + 'snapshot.pth'
    torch.save(snapshot_model, savepath)
    savepath = result_dir + result_name + 'trained.pth'
    torch.save(forward_model, savepath)
    
    savepath = result_dir + result_name + 'training_results.pdf'
    plt.figure(figsize=(9,5))
    x = range(len(train_losses))
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.plot(x, train_losses, label='Train')
    plt.plot(x, val_losses, label='Valid')
    plt.plot([0, snapshot_epoch, len(train_losses)-1], [val_losses[0], val_losses[snapshot_epoch], val_losses[len(train_losses)-1]], 
             marker='*', ls='None', markersize=15, c='k')
    plt.legend()
    plt.savefig(savepath)
    return


def generate_melody(device, valid_dataset, sample_idx, model, predict_length, interval_range, filename, start_pitch=74):
    """
    Lets model generate a melody of specified length
    :param device: 
    :param valid_dataset: validation dataset for starter intervals
    :param sample_idx: index of elem in dataset that is used
    :param model: 
    :param predict_length: 
    :param interval_range: 
    :param filename: 
    :param start_pitch: 
    :return:
    """
    interval_values = np.arange(interval_range[0], interval_range[1]+1, 1)
    generated_track = []
    x, y = valid_dataset[sample_idx]
    x = x.to(device)
    # correct for normalization in dataset
    x_orig = np.array(x.cpu().detach().numpy() * max(abs(interval_range[0]), interval_range[1]), dtype=int)
    
    model.eval()
    generated_track = list(x_orig)
    for i in range(predict_length):
        output = model(x)
        chosen_idx = np.argmax(output.cpu().detach().numpy())
        chosen_interval = interval_values[chosen_idx]
        generated_track.append(chosen_interval)
        x_new = np.array(generated_track[-len(x):])
        x = torch.tensor(x_new / max(abs(interval_range[0]), interval_range[1]), dtype=torch.float32)
        x = x.to(device)

    # translate from sequence of intervals to sequence of pitches
    track = [start_pitch]
    for interval in generated_track:
        track.append(track[-1] + interval)
    for i, note in enumerate(track):
        # clamp to midi-valid range
        if note < 1:
            track[i] = 1
        if note > 255:
            track[i] = 255
    track = np.array(track)
    numpy_notes = midi_utils.prediction_to_numpy(track, 1024)
    
    # create midi track
    new_track = midi_utils.numpy_to_midi_track(numpy_notes, 1, 'Modified')
    # make new song with the new track
    new_track_dict = {}
    new_track_dict['0'] = ['0, 0, Header, 1, 1, 1024\n', '0, 0, End_of_file']
    new_track_dict['1'] = new_track
    csv_list = midi_utils.track_dict_to_csv(new_track_dict)
    midi_utils.write_to_midi(csv_list, filename)
    return 
