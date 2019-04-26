import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def get_dataset_representation_from_tracks(tracks, feature_qty=10, prediction_qty=1):

    x = []
    y = []

    for track in tracks:
        track_x, track_y = get_dataset_representation_from_track(track, feature_qty, prediction_qty)
        x.extend(track_x)
        y.extend(track_y)

    return x, y


def get_dataset_representation_from_track(track, feature_qty=10, prediction_qty=1):

    x = []
    y = []
    if len(track) < feature_qty + prediction_qty:
        raise Exception("Track is shorter than {0} the number of features for data set representation.".format(feature_qty + prediction_qty))

    sample_qty = len(track) - prediction_qty - feature_qty  # number of samples we can generate from one track if we always increment by one tone

    for first_note_index in range(sample_qty):
        features = track[first_note_index:first_note_index + feature_qty]
        prediction = track[first_note_index + feature_qty: first_note_index + feature_qty + prediction_qty]
        x.append(features)
        y.append(prediction)

    return np.array(x), np.array(y)


class TrackDataset(Dataset):
    def __init__(self, x, y):
        super(TrackDataset, self).__init__()
        self.x = x
        self.y = y

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
        x = x[:, 0] # take only the pitch
        x = torch.tensor(x, dtype=torch.float32)

        y = self.y[index]
        y = y[:, 0] # take only the pitch
        y = torch.tensor(y, dtype=torch.float32)
        return x, y
