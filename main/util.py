import os
import numpy as np
import pandas as pd
import torch
# requires: pip install PySoundFile to avoid RuntimeErr: No audio I/O backend is available.
# https://stackoverflow.com/questions/62543843/cannot-import-torch-audio-no-audio-backend-is-available
import torchaudio
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm

torch.set_default_dtype(torch.float32)


def loadCSV(filepath: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    :param str filepath:
    :rtype: tuple[torch.Tensor, torch.Tensor, int]
    """
    features_df = pd.read_csv(filepath)

    # Remove broken jazz.00054.wav
    features_df.drop(features_df[features_df['filename'] == 'jazz.00054.wav'].index, inplace=True)
    
    # Get the number of items in the features
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

        self.x = []
        self.y = []
        # Assume sample rates of all the files are the same
        _, self.sample_rate = torchaudio.load(open(f'{path}/{os.listdir(path)[0]}/{os.listdir(path + "/" + os.listdir(path)[0])[0]}', 'rb'))
        resampler = torchaudio.transforms.Resample(self.sample_rate, self.sample_rate // 4)
        with tqdm(total=1000) as pbar:
            for genre in os.listdir(path):
                current_onehot = torch_onehot(cats, genre)
                for file in os.listdir(f'{path}/{genre}'):
                    if file == 'jazz.00054.wav':
                        continue
                    current_x, _ = torchaudio.load(open(f'{path}/{genre}/{file}', 'rb'))

                    current_x = resampler(current_x)[:, 0:22050].swapaxes(0, 1)
                    self.x.append(current_x)
                    self.y.append(current_onehot[0])
                    pbar.update(1)

        self.device = device

    def __getitem__(self, item):
        return self.x[item].to(self.device), self.y[item].to(self.device)

    def __len__(self):
        return len(self.x)


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

        self.x = []
        self.y = torch.zeros((1, len(cats)))
        for genre in os.listdir(path):
            current_onehot = torch_onehot(cats, genre)
            for file in os.listdir(f'{path}/{genre}'):
                if file == 'jazz00054.png':
                    continue
                self.x.append(f'{path}/{genre}/{file}')
                self.y = torch.concat((self.y, current_onehot))

        self.x = self.x[1:]
        self.y = self.y[1:]
        self.device = device

    def __getitem__(self, item):
        return read_image(self.x[item]).type(torch.FloatTensor).to(self.device), self.y[item].to(self.device)

    def __len__(self):
        return len(self.x)


class MultiSourceData(Dataset):
    def __init__(self, path: str, device: str = 'cpu'):
        self.png = PngData(f'{path}/images_original', device=device)
        self.wav = WavData(f'{path}/genres_original', device=device)
        self.csv = loadCSV(f'{path}/features_30_sec.csv')

    def __getitem__(self, item):
        png, label = self.png[item]
        wav = self.wav[item][0]
        csv = self.csv[0][item]

        return png, wav, csv, label

    def __len__(self):
        return len(self.png)
