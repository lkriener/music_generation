import src.midi_utils as midi_utils
import pygame
from pypianoroll import Multitrack, Track
import pypianoroll
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import glob 
from copy import deepcopy

# Check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')

    
    



def get_semitones_to_C(csv_string):
    """
    Get the number semitones to C major 

    can be different for each track
    therefore this function operates on a single track from the track_dict
    info does not have to be included in track
    in that case use 0 semitones
    return value, bool if default was used
    :param csv_string:
    :return:
    """

    default = True
    semitones = 0 # default
    for line in csv_string:
        # find line with Time_signature
        split_line = [x.strip() for x in line.split(',')]
        if split_line[2] == 'Key_signature':
            semitones = int(split_line[3])
            default = False
            break
    return semitones, default




def get_track(midi_filename, voice, beat_resolution, transpose = True):
    csv_text = midi_utils.load_to_csv(midi_filename)
    # get semitones to C major 
    semitones,_ = get_semitones_to_C(csv_text)
    # get multitrack object from midi 
    multitrack = pypianoroll.parse(midi_filename, beat_resolution=beat_resolution)
    # get the voice track 
    if transpose: 
        track = pypianoroll.transpose(multitrack.tracks[voice], -semitones)
    else:
        track = multitrack.tracks[voice]
    return track




def get_all_pianorolls(voice, home_dir, beat_resolution=4):
    '''
    Returns a large concatenated pianoroll from all Bach Chorals midi files
    :voice: 0 = soprano, 1 = alto, 2 = tenor, 3 = bass 
    :home_dir: home directory we extract midi files from
    :beat_resolution: minimal pianoroll time step. Default 4=sixteenth
    :return:
    '''
    list_pianorolls = []
    midi_files = [] # store all midi files 
    os.chdir(home_dir + "/data/raw/bach")  # go to a folder relative to home dir
  
    for midi_file in glob.glob("*.mid"):
        track = get_track(midi_file, voice, beat_resolution=beat_resolution)
        # get the flattened representation of pianoroll
        pianoroll_flattened = flatten_one_hot_pianoroll(track.pianoroll)
        # add it to the global list of all tracks
        list_pianorolls.append(pianoroll_flattened)
        
        midi_files.append(midi_file)
    
    # convert into an array 
    all_pianorolls = np.concatenate(list_pianorolls)
    
    return all_pianorolls, midi_files



def get_extremum_pitches(list_pianorolls):
    '''
    Return extremum pitches and pitch dictionnary of a given pitch sequence
    :list_pianorolls: list containing pianorolls sequence
    :return:
    '''
    list_min = []
    list_max = []
    
    for pianoroll in list_pianorolls:
        minimum  = np.min(pianoroll[np.nonzero(pianoroll)])
        maximum = np.max(pianoroll)
        list_min.append(minimum)
        list_max.append(maximum)
    global_lower = int(min(list_min))
    global_upper = int(max(list_max))
    n_notes = global_upper - global_lower + 2 
    
    return global_lower, global_upper, n_notes
    
    
    
# Declaring the train method
def train(net, data, data2=None, mode="melody_generation", epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' Training a network 

        net: NoteRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss
    
    '''
    net.train()
    
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    if mode == "harmonization":
        data2, val_data2 = data2[:val_idx], data2[val_idx:]
    
    if(train_on_gpu):
        net.cuda()
    
    train_losses = []
    val_losses = []
    
    # parameters for early stopping
    ### Early stopping code
    best_val_loss = np.inf
    best_net = None
    patience = 5 # if no improvement after 5 epochs, stop training
    counter = 0
    best_epoch = 0
        
    
    n_notes = net.n_notes 
    for e in range(epochs):
        
        store_losses = []
        
        # initialize hidden state
        h = net.init_hidden(batch_size)
        
        if mode == "melody_generation":
            batch_generator = get_pianoroll_batches(data, batch_size, seq_length)
        elif mode == "harmonization":
            batch_generator = get_pianoroll_batches_harmonization(data, data2, batch_size, seq_length)
        for x, y in batch_generator:
            
            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode_batch(x, n_notes)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()
            
            # get the output from the model
            output, h = net(inputs, h)
            # calculate the loss and perform backprop
            loss = criterion(output, targets.contiguous().view(batch_size*seq_length).long())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            store_losses.append(loss.item())
            
        train_loss = np.mean(store_losses)
        train_losses.append(train_loss)
        
        # Get validation loss
        val_h = net.init_hidden(batch_size)
        net.eval()

        store_losses = []

        if mode == "melody_generation":
            batch_generator_val = get_pianoroll_batches(val_data, batch_size, seq_length)
        elif mode == "harmonization":
            batch_generator_val = get_pianoroll_batches_harmonization(val_data, val_data2, batch_size, seq_length)
            
        for x, y in batch_generator_val:
            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode_batch(x, n_notes)
            x, y = torch.from_numpy(x), torch.from_numpy(y)

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            val_h = tuple([each.data for each in val_h])

            inputs, targets = x, y
            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            output, val_h = net(inputs, val_h)
            loss = criterion(output, targets.contiguous().view(batch_size*seq_length).long())

            store_losses.append(loss.item())

        net.train() # reset to train mode after iterationg through validation data

        
        val_loss = np.mean(store_losses)
        val_losses.append(val_loss)


        print("Epoch: {}/{}...".format(e+1, epochs),
              "Loss: {:.4f}...".format(train_loss),
              "Val Loss: {:.4f}".format(val_loss))
    
        ### Early stopping code, does not stop training but copy best model 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_net = deepcopy(net)
            best_epoch = e
            counter = 0
        else:
            counter += 1
        if counter == patience:
            print('No improvement for {} epochs; training stopped.'.format(patience))
        ###

                  
    return np.array(train_losses), np.array(val_losses), best_net, best_epoch
                

    
def display_losses(ax, train_losses, val_losses, best_epoch):
    n_epochs = len(train_losses)
    epochs = np.arange(n_epochs)
    ax.plot(epochs, train_losses, label='Train')
    ax.plot(epochs, val_losses, label="Valid")
    ax.legend(loc = 'best')
    ax.plot([0, best_epoch, n_epochs-1], [val_losses[0], val_losses[best_epoch], val_losses[n_epochs-1]], marker='*', ls='None', markersize=15, c='k')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Cross Entropy Loss")
                
                
                
# Defining a method to generate the next character
def predict(net, note, h=None):
        ''' Given a note, predict the next note.
            Returns the predicted note and the hidden state.
        '''
        
        # tensor inputs
        x = np.array([[note]])
        x = one_hot_encode_batch(x, net.n_notes)
        inputs = torch.from_numpy(x)
        
        if(train_on_gpu):
            inputs = inputs.cuda()
       
        # detach hidden state from history
        h = tuple([each.data for each in h])
        # get the output of the model
        out, h = net(inputs, h)
        

        # get the character probabilities
        p = F.softmax(out, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
        
        note_range = np.arange(net.n_notes)
        # select the likely next note with some element of randomness
        p = p.numpy().squeeze()
        note = np.random.choice(note_range, p=p/p.sum())
        
        # return the encoded value of the predicted char and the hidden state
        return note, h
                
                

# Declaring a method to generate new melody
def sample(net, size, prime=[10,10,12,12]):
        
    if(train_on_gpu):
        net.cuda()
    else:
        net.cpu()
    
    net.eval() # eval mode
    
    # First off, run through the prime notes
    notes = [no for no in prime]
    h = net.init_hidden(1)
    for no in prime:
        note, h = predict(net, no, h)
    notes.append(note)
    
    # Now pass in the previous note and get a new one
    for ii in range(size):
        note, h = predict(net, notes[-1], h)
        notes.append(note)

    return np.array(notes)





def sample_harmonization(net, seq, prime):
        
    if(train_on_gpu):
        net.cuda()
    else:
        net.cpu()
    
    net.eval() # eval mode
    
    size = len(seq)
    
    # First off, run through the prime notes
    notes = [no for no in prime]
    h = net.init_hidden(1)
    for no in prime:
        note, h = predict(net, no, h)
    notes.append(note)
    
    # Now pass in the previous character and get a new one
    for ii in range(5,size):
        note, h = predict(net, seq[ii], h)
        notes.append(note)

    return np.array(notes)


def one_hot_encode_batch(flattened_pianoroll, n_notes):
    """
    Return a one-hot batch to perform RNN training
    :flattened_pianoroll:
    :n_notes:
    :return:
    """
    
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*flattened_pianoroll.shape), n_notes), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), flattened_pianoroll.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*flattened_pianoroll.shape, n_notes))
    
    return one_hot


def one_hot_encode_pianoroll(flattened_pianoroll, n_notes):
    """
    Converts a flattened pianoroll to a one-hot matrix
    keeping 0 for the silences
    :flattened_pianoroll:
    :n_notes: range of notes including the silence
    :return:
    """
    one_hot = np.zeros((len(flattened_pianoroll), n_notes))
    for i in range(len(flattened_pianoroll)):
        if flattened_pianoroll[i] > 0: # if it is a note, and not a silence
            one_hot[i,flattened_pianoroll[i]] = 1
    return one_hot



def flatten_one_hot_pianoroll(one_hot_pianoroll):
    """
    Returns a flattened piano_roll array 
    :param one_hot_pianoroll: pianoroll representation from track object
    :return:
    """
    flattened_pianoroll = np.argmax(one_hot_pianoroll, axis=1)
    return flattened_pianoroll
    
def scale_pianoroll(flattened_pianoroll, global_lower):
    """
    Scales flattened pianoroll to values near 0
    keep 0 for silences
    :flattened_pianoroll: 
    :global_lower: lower pitch of the whole tracks dataset 
    :return:
    """
    scaled_pianoroll = np.copy(flattened_pianoroll - global_lower + 1)
    scaled_pianoroll[np.where(scaled_pianoroll<0)] = 0 
    return scaled_pianoroll

def unscale_pianoroll(scaled_pianoroll, global_lower):
    """
    Returns the pianoroll in the initial config.
    before applying scale_pianoroll
    :scaled_pianoroll:
    :global_lower:
    :return:
    """
    unscaled_pianoroll = np.copy(scaled_pianoroll + global_lower - 1)
    unscaled_pianoroll[np.where(unscaled_pianoroll == global_lower-1)] = 0 # reset the silences to 0
    return unscaled_pianoroll 



# Defining method to make mini-batches for training
def get_pianoroll_batches(arr, batch_size, seq_length):
    '''
    Create a generator that returns batches of size
    batch_size x seq_length from arr.
       
    :arr: Array you want to make batches from
    :batch_size: Batch size, the number of sequences per batch
    :seq_length: Number of encoded notes in a sequence
    '''
    
    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr)//batch_size_total
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    
    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

        

def get_pianoroll_batches_harmonization(arr, arr2, batch_size, seq_length):
    '''
    Create a generator that returns batches of size
    batch_size x seq_length from arr.
       
    :arr: Array you want to make batches from
    :batch_size: Batch size, the number of sequences per batch
    :seq_length: Number of encoded notes in a sequence
    '''
    
    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr)//batch_size_total
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    arr2 = arr2[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    arr2 = arr2.reshape((batch_size, -1))
    # iterate through the array, one sequence at a time
    
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        y = arr2[:, n:n+seq_length]
        
        yield x, y


def process_harmonization(midi_filename, net, global_lower, real_tracks, voice_togenerate):
    soprano_pianoroll = scale_pianoroll(flatten_one_hot_pianoroll(real_tracks[0].pianoroll), global_lower)
    pianoroll_real = scale_pianoroll(flatten_one_hot_pianoroll(real_tracks[voice_togenerate].pianoroll), global_lower)
    # predict the second voice
    pianoroll = sample_harmonization(net[voice_togenerate], soprano_pianoroll, prime = pianoroll_real[:4])
    # go back to the pitch range 
    pianoroll = unscale_pianoroll(pianoroll, global_lower)
    # convert to one-hot representation for track object
    one_hot = one_hot_encode_pianoroll(pianoroll, 128)*90
    # get the track object
    track = Track(pianoroll=one_hot, name='generated track')
    
    return track

def process_single(net, start, seq_size, global_lower, beat_resolution=2):
    notes = sample(net, seq_size, prime=[start,start,start+2,start+2])
    # put back to the pitch ranges
    pianoroll = unscale_pianoroll(notes, global_lower) 
    # create a one_hot_pianoroll
    one_hot = one_hot_encode_pianoroll(pianoroll, 128)*90
    # store it a in a track object
    track = Track(pianoroll=one_hot, name='new track')
    # create a multitrack made of the generated track object
    multitrack = Multitrack(tracks=[track], tempo = 90, beat_resolution=beat_resolution)
    
    return multitrack



# Class for the model 

# Declaring the model
class NoteRNN(nn.Module):
    
    def __init__(self, n_notes, n_hidden=256, n_layers=2,
                               drop_prob=0.2, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        self.n_notes = n_notes 
        #define the LSTM
        self.lstm = nn.LSTM(self.n_notes, n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        #define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        #define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, self.n_notes)
      
    
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
                
        #get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)
        
        #pass through a dropout layer
        out = self.dropout(r_output)
        
        # Stack up LSTM outputs using view
        out = out.contiguous().view(-1, self.n_hidden)
        
        #put x through the fully-connected layer
        out = self.fc(out)
        
        # return the final output and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden




