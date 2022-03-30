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
        cats = sorted(os.listdir(path))  # cats = categories

        self.x = torch.zeros((1, 1))
        self.y = torch.zeros((1, len(cats)))
        for genre in os.listdir(path):
            current_onehot = self._onehot(cats, genre)
            for file in os.listdir(f'{path}{os.sep}{genre}'):
                # Assume sample rates of all the files are the same
                current_x, self.sample_rate = torchaudio.load(open(f'{path}/{genre}/{file}', 'rb'))

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

        self.x = self.x[1:]
        self.y = self.y[1:]
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
            result = torch.zeros(1, len(cats))
            result[0, i] = 1
            return result

    def __getitem__(self, item):
        return self.x[item].to(self.device), self.y[item].to(self.device)

    def __len__(self):
        return self.x.shape[0]
