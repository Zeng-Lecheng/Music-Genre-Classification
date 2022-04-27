import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle as pkl

import util
from rnn import RNNet
from rcnn import RcnnNet
from dnn import DeepNeuralNet
from util import WavData

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

device = 'cpu'  # in case nvidia GPU is available, change to 'cuda' might accelerate



def get_data():
    feat_data, feat_labels, length = util.loadCSV(f'../data/features_30_sec.csv')
    png_data = pkl.load(open('png_data.pkl', 'rb'))     #PngData
    wav_data = WavData('../data/genres_original', device=device)
    loader = DataLoader(png_data, length)
    png_inputs, png_labels = next(iter(loader))
    loader = DataLoader(wav_data, length)
    wav_inputs, wav_labels = next(iter(loader))

    for i in tqdm(range(0, 999)):
        pass


def model_train(train: bool, verbose: bool):
    dnn = DeepNeuralNet().to(device)
    dnn.load_state_dict(torch.load('../saved_models/dnn.pt'))
    cnn = RcnnNet().to(device)
    cnn.load_state_dict(torch.load('../saved_models/rcn.pt'))
    rnn = RNNet().to(device)
    rnn.load_state_dict(torch.load('../saved_models/rnn_with_cov_final.pt'))




if __name__ == '__main__':
    model_train()
