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

# use cpu by default, change to cuda if you want and change it back before committing
# we only debug and ensure everything works well on cpu
device = 'cpu'


def get_data():
    # Loading in feat_inputs: shape = torch.Size([999, 58])
    feat_inputs, feat_labels, length = util.loadCSV(f'../data/features_30_sec.csv')

    # Loading in the saved Deep Neural Net
    dnn = DeepNeuralNet().to(device)
    dnn.load_state_dict(torch.load('../saved_models/dnn.pt'))

    # Running all inputs through the DeepNeuralNet to get outputs: shape = torch.Size([999, 10])
    feat_outputs = dnn(feat_inputs)

    # Loading in wav_inputs (shape = torch.Size([999, 900, 1])) and labels
    wav_data = pkl.load(open('../data/wav_data.pkl', 'rb'))  # WavData
    wav_loader = DataLoader(wav_data, length)
    wav_inputs, wav_labels = next(iter(wav_loader))

    # Loading in the saved Recurrent Neural Net
    rnn = RNNet().to(device)
    rnn.load_state_dict(torch.load('../saved_models/rnn_with_cov_final.pt'))

    # Running all inputs through the RNNet to get outputs: shape = torch.Size([999, 10])
    wav_outputs = rnn(wav_inputs)

    print(wav_outputs.shape)

    png_data = pkl.load(open('../data/png_data.pkl', 'rb'))  # PngData
    png_loader = DataLoader(png_data, length)
    png_inputs, png_labels = next(iter(png_loader))

    cnn = RcnnNet().to(device)
    cnn.load_state_dict(torch.load('../saved_models/rcn.pt'))


def model_train(verbose: bool):
    get_data()


if __name__ == '__main__':
    model_train(True)
