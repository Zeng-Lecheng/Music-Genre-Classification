import os

import torch
import torchaudio
from torch.utils.data import Dataset

torch.set_default_dtype(torch.float32)


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
        cats = sorted(os.listdir(path)) #cats = categories

        x = []
        y = []
        for genre in os.listdir(path):
            onehot = self._onehot(cats, genre)
            for file in os.listdir(f'{path}{os.sep}{genre}'):
                # Assume sample rates of all the files are the same
                current_x, self.sample_rate = torchaudio.load(open(f'{path}{os.sep}{genre}{os.sep}{file}', 'rb'))
                x.append(current_x)
                y.append(onehot)
        self.x = torch.concat(x)
        self.y = torch.concat(y)
        self.device = device

    @staticmethod
    def _onehot(cats: list[str], key: str) -> torch.Tensor:
        """
        Generate the onehot tensor of given categories and key.
        Example
        -------
            cats: ['apple', ''banana', 'orange']
            key: ['banana']
        Return:
            torch.Tensor([0., 1., 0.])
        :param cats: list of categories
        :param key: key
        :return: tensor of onehot representation
        """
        try:
            i = cats.index(key)
        except IndexError as e:
            raise IndexError(f'Error reading genres. The genre ({key}) I got is not in the list ({cats}). ')
        else:
            result = torch.zeros(len(cats))
            result[i] = 1
            return result

    def __getitem__(self, item):
        return self.x[item].to(self.device), self.y[item].to(self.device)

    def __len__(self):
        return self.x.shape[0]
