import os
import numpy as np
from pathlib import Path
from towhee import pipeline
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import util


def main():
    data = DataLoader(util.WavData('../data/genres_original'))


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

def getVGGishEmbeddings(filepath = '../data/vggish'):
    embedding_pipeline = pipeline('towhee/audio-embedding-vggish')
    dataset_path = filepath
    music_list = [f for f in Path(dataset_path).glob('*')]
    vec_sets = []
    with tqdm(total=999) as progressbar:
        for audio_path in music_list:
            vecs = embedding_pipeline(str(audio_path))
            norm_vecs = [vec / np.linalg.norm(vec) for vec in vecs[0][0]]
            vec_sets.append(norm_vecs)
            progressbar.update(1)
    print(len(vec_sets))
    print(vec_sets)
    vectors = []
    for i in tqdm(range(len(vec_sets))):
        for vec in vec_sets[i]:
            vectors.append(vec)
    print(vectors)
    with open('embeddings.txt', 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(vectors)

