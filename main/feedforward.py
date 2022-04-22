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

device = 'cpu'


def model_train():
    dnn = DeepNeuralNet().to(device)
    rcnn = RcnnNet()
    rnn = RNNet()

    rcnn.load_state_dict(torch.load('../saved_models/rnn_with_cov_final.pkl'))


if __name__ == '__main__':
    model_train()
