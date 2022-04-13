import numpy as np
import matplotlib.pyplot as plt
# import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import util


def main():
    data = DataLoader(util.WavData('../data/genres_original'))
    x = next(iter(data))
    print(x.shape)


def model():
    pass

def test():
    pass

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Initialize the layers


    def forward(self, x):
        """
        :param torch.Tensor x:
        :return:
        """

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Initialize the layers


    def forward(self, x):
        """
        :param torch.Tensor x:
        :return:
        """
        x = F.relu(self.ffLayer1(x))
        x = torch.sigmoid(self.ffLayer2(x))
        return x


class AttentionDecoder(Decoder):
    def __init__(self):
        super(Decoder, self).__init__()
        # Initialize the layers


    def forward(self, x):
        """
        :param torch.Tensor x:
        :return:
        """

        return x

main()