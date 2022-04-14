import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.io import read_image

torch.set_default_dtype(torch.float32)


def loadCSV(filepath: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    :param str filepath:
    :rtype: tuple[torch.Tensor, torch.Tensor, int]
    """
    features_df = pd.read_csv(filepath)
    length = len(features_df.index)

    # First drop the file name, it is not needed.
    del features_df['filename']

    # Save the names of the categories
    cats = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

    # First save the label column to a separate variable, which can then be turned into onehot Tensor.
    labels_df = features_df['label']
    labels = makeOnehotTensorFromDataframe(cats, labels_df)

    # Then drop the label column and turn the features into a Tensor.
    del features_df['label']
    # normalization
    for col in features_df.columns:
        features_df[col] = (features_df[col] - min(features_df[col])) / (max(features_df[col]) - min(features_df[col]))
    features = torch.as_tensor(features_df.values, dtype=torch.float32)
    return features, labels, length


def makeOnehotTensorFromNdarray(cats: list[str], keys: np.ndarray) -> torch.Tensor:
    labels_list = []
    n = len(keys)
    for i in range(0, n):
        key = keys[n]
        result = makeOnehot(cats, key, i)
        labels_list.append(result)
    labels = torch.as_tensor(labels_list, dtype=torch.float32)
    return labels


def makeOnehotTensorFromDataframe(cats: list[str], keys: pd.DataFrame) -> torch.Tensor:
    labels_list = []
    total_cats = len(cats)
    index = 0
    for key in keys:
        result = makeOnehot(cats, total_cats, key, index)
        labels_list.append(result)
        index += 1
    labels = torch.as_tensor(labels_list, dtype=torch.float32)
    return labels


def makeOnehot(cats: list[str], total_cats: int, key: str, index: int) -> list[int]:
    try:
        i = cats.index(key)
    except IndexError as e:
        raise IndexError(f'The genre ({key}) I got is not in the list ({cats}).')
    else:
        result = [0] * total_cats
        result[i] = 1
        return result


def torch_onehot(cats: list[str], key: str) -> torch.Tensor:
    """
    Generate the onehot tensor of given categories and key.
    Example
    -------
        cats: ['apple', ''banana', 'orange']
        key: ['banana']
    Return:
        torch.Tensor([[0., 1., 0.]])
    :param cats: list of categories
    :param key: key
    :return: tensor of onehot representation
    """
    try:
        i = cats.index(key)
    except IndexError:
        raise IndexError(f'Error reading genres. The genre ({key}) I got is not in the list ({cats}). ')
    else:
        result = torch.zeros(1, len(cats))
        result[0, i] = 1
        return result


class WavData(Dataset):
    def __init__(self, path: str, device: str = 'cpu'):
        """
        Assume directory structure be like:
        data
        |- genre_1
            |- genre_1.00000.wav
            |- genre_2.00000.wav
            ...
        |- genre_2
            |- ...
        |- genre_3
            |- ...
        ...

        Then path would be 'data'
        :param path: Path to the root of all genres
        :param device: Where the data is going to be sent to ('cpu' or 'cuda')
        """
        cats = sorted(os.listdir(path))  # cats = categories

        self.x = torch.zeros((1, 1))
        self.y = torch.zeros((1, len(cats)))
        for genre in os.listdir(path):
            current_onehot = torch_onehot(cats, genre)
            for file in os.listdir(f'{path}/{genre}'):
                # Assume sample rates of all the files are the same
                current_x, self.sample_rate = torchaudio.load(open(f'{path}/{genre}/{file}', 'rb'))

                current_x = current_x[:, :900]     # too large, cut it short
                # Do zero padding if size does not match
                if current_x.shape[1] > self.x.shape[1]:
                    pad_self_x = torch.zeros((self.x.shape[0], current_x.shape[1]))
                    pad_self_x[:, :self.x.shape[1]] = self.x
                    self.x = torch.concat((pad_self_x, current_x))
                elif current_x.shape[1] < self.x.shape[1]:
                    pad_current = torch.zeros(1, self.x.shape[1])
                    pad_current[:, :current_x.shape[1]] = current_x
                    self.x = torch.concat((self.x, pad_current))
                else:
                    self.x = torch.concat((self.x, current_x))

                self.y = torch.concat((self.y, current_onehot))

        self.x = self.x[1:, :, None]
        self.x = self.x / (self.x.max() - self.x.min())     # normalization
        self.y = self.y[1:]
        self.device = device

    def __getitem__(self, item):
        return self.x[item].to(self.device), self.y[item].to(self.device)

    def __len__(self):
        return self.x.shape[0]


class PngData(Dataset):
    def __init__(self, path: str, device: str = 'cpu'):
        """
        Assume directory structure be like:
        data
        |- genre_1
            |- genre_1.00000.png
            |- genre_2.00000.png
            ...
        |- genre_2
            |- ...
        |- genre_3
            |- ...
        ...

        Then path would be 'data'
        :param path: Path to the root of all genres
        :param device: Where the data is going to be sent to ('cpu' or 'cuda')
        """
        cats = sorted(os.listdir(path))

        shape = read_image(f'{path}/{cats[0]}/{os.listdir(f"{path}/{cats[0]}")[0]}')[None, :].shape

        self.x = torch.zeros(shape)
        self.y = torch.zeros((1, len(cats)))
        for genre in os.listdir(path):
            current_onehot = torch_onehot(cats, genre)
            for file in os.listdir(f'{path}/{genre}'):
                current_x = read_image(f'{path}/{genre}/{file}')[None, :]
                self.x = torch.concat((self.x, current_x))
                self.y = torch.concat((self.y, current_onehot))

        self.x = self.x[1:]
        self.y = self.y[1:]
        self.device = device

    def __getitem__(self, item):
        return self.x[item].to(self.device), self.y[item].to(self.device)

    def __len__(self):
        return self.x.shape[0]
