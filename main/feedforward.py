import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from rnn import RNNet
from rcnn import RcnnNet
from dnn import DeepNeuralNet

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

device = 'cpu'  # in case nvidia GPU is available, change to 'cuda' might accelerate


def model_train(train: bool, verbose: bool):
    dnn = DeepNeuralNet().to(device)
    dnn.load_state_dict(torch.load('../saved_models/dnn.pt'))
    cnn = RcnnNet().to(device)
    cnn.load_state_dict(torch.load('../saved_models/rcn.pt'))
    rnn = RNNet().to(device)
    rnn.load_state_dict(torch.load('../saved_models/rnn_with_cov_final.pt'))




if __name__ == '__main__':
    model_train()
