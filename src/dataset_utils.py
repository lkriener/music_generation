import numpy as np
import torch
from torch.utils.data import Dataset


def get_dataset_representation_from_tracks(tracks, feature_qty=10, prediction_qty=1, intervals=False):
    """
    Get dataset in moving window representation from a list of tracks.
    :param tracks:
    :param feature_qty:
    :param prediction_qty:
    :return:
    """

    x = []
    y = []

    for track in tracks:
        if intervals:
            track_x, track_y = get_dataset_representation_from_track_intervals(track, feature_qty, prediction_qty)
        else:
            track_x, track_y = get_dataset_representation_from_track(track, feature_qty, prediction_qty)
        x.extend(track_x)
        y.extend(track_y)

    return x, y


def get_dataset_representation_from_track(track, feature_qty=10, prediction_qty=1):
    """
    Get dataset in moving window representation from a single track.
    :param track:
    :param feature_qty:
    :param prediction_qty:
    :return:
    """

    x = []
    y = []
    if len(track) < feature_qty + prediction_qty:
        raise Exception("Track is shorter than the moving window size {0} for dataset representation.".format(feature_qty + prediction_qty))

    sample_qty = len(track) - prediction_qty - feature_qty  # number of samples we can generate from one track if we always increment by one tone

    for first_note_index in range(sample_qty):
        features = track[first_note_index:first_note_index + feature_qty]
        prediction = track[first_note_index + feature_qty: first_note_index + feature_qty + prediction_qty]
   
        x.append(features)
        y.append(prediction)
        
    return np.array(x), np.array(y)


def get_dataset_representation_from_track_intervals(track, feature_qty=10, prediction_qty=1):
    """
    Get dataset in moving window representation from a single track in interval formulation
    :param track:
    :param feature_qty:
    :param prediction_qty:
    :return:
    """

    x = []
    y = []
    if len(track) < feature_qty + prediction_qty:
        raise Exception("Track is shorter than the moving window size {0} for dataset representation.".format(feature_qty + prediction_qty))

    sample_qty = len(track) - prediction_qty - feature_qty - 2  # number of samples we can generate from one track if we always increment by one tone

    for first_note_index in range(sample_qty):
        # features
        pitches = track[first_note_index:first_note_index + feature_qty + 1][:,0]
        lengths = track[first_note_index:first_note_index + feature_qty + 1][:,1]
        intervals = np.diff(pitches)
        features = np.dstack((intervals, lengths[:-1]))[0]
        # prediction
        pitches = track[first_note_index + feature_qty: first_note_index + feature_qty + prediction_qty + 1][:,0]
        lengths = track[first_note_index + feature_qty: first_note_index + feature_qty + prediction_qty + 1][:,1]
        intervals = np.diff(pitches)
        prediction = np.dstack((intervals, lengths[:-1]))[0]
   
        x.append(features)
        y.append(prediction)
        
    return np.array(x), np.array(y)



class TrackDataset(Dataset):
    def __init__(self, x, y, drop_length=False, intervals=False):
        super(TrackDataset, self).__init__()

        self.x = x
        self.y = y
        self.intervals = intervals
        self.drop_length = drop_length

    def __len__(self):
        """
        Denotes the total number of samples.
        :return:
        """

        return len(self.x)

    def __getitem__(self, index):
        """
         Generates one sample of data.
         :return:
        """

        # Select sample
        x = self.x[index]
        if self.drop_length:
            x = x[:, 0]
        x = torch.tensor(x, dtype=torch.float32)

        y = self.y[index]
        if self.drop_length:
            y = y[:, 0]
        if self.intervals:
            y = torch.tensor(y, dtype=torch.long)
        else:
            y = torch.tensor(y, dtype=torch.float32)
        return x, y
