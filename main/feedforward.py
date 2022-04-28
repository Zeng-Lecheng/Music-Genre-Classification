from math import floor, ceil
from statistics import mean
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
torch.manual_seed(0)
device = 'cpu'


def get_data():
    # Loading in feat_inputs: shape = torch.Size([999, 58])
    feat_inputs, feat_labels, length = util.loadCSV(f'../data/features_30_sec.csv')

    # Loading in the saved Deep Neural Net
    dnn = DeepNeuralNet().to(device)
    dnn.load_state_dict(torch.load('../saved_models/dnn.pt'))

    # Running all inputs through the DeepNeuralNet to get outputs: shape = torch.Size([999, 10])
    with torch.no_grad():
        feat_outputs = dnn(feat_inputs)

    # Loading in png_inputs: shape = torch.Size([999, 4, 288, 432])
    png_data = pkl.load(open('../data/png_data.pkl', 'rb'))  # PngData
    png_loader = DataLoader(png_data, length)
    png_inputs, png_labels = next(iter(png_loader))

    # Loading in the saved Residual Convolutional Neural Net
    cnn = RcnnNet().to(device)
    cnn.load_state_dict(torch.load('../saved_models/rcn.pt'))

    # Running all inputs through the RcnnNet to get outputs: shape = torch.Size([999, 10])
    with torch.no_grad():
        png_outputs = cnn(png_inputs)

    # Loading in wav_inputs (shape = torch.Size([999, 900, 1])) and labels
    wav_data = pkl.load(open('../data/wav_data.pkl', 'rb'))  # WavData
    wav_loader = DataLoader(wav_data, length)
    wav_inputs, wav_labels = next(iter(wav_loader))

    # Loading in the saved Recurrent Neural Net
    rnn = RNNet().to(device)
    rnn.load_state_dict(torch.load('../saved_models/rnn_with_cov_final.pt', map_location=torch.device('cpu')))

    # Running all inputs through the RNNet to get outputs: shape = torch.Size([999, 10])
    with torch.no_grad():
        wav_outputs = rnn(wav_inputs)

    # Convert to numpy arrays to make absolutely sure there's no autograd
    feat_outputs = feat_outputs.detach().numpy()
    png_outputs = png_outputs.detach().numpy()
    wav_outputs = wav_outputs.detach().numpy()

    # Make sure the outputs are all in the same order
    if not confirmOrder(torch.argmax(feat_labels, dim=1), torch.argmax(png_labels, dim=1), torch.argmax(wav_labels, dim=1)):
        raise Exception("At least one of the datasets is out of order with the others.")

    # Concatenate the outputs to form the inputs for the feedforward neural network: shape = torch.Size([999, 30])
    ff_inputs = np.concatenate((feat_outputs, png_outputs, wav_outputs), axis=1)
    ff_inputs = torch.from_numpy(ff_inputs)

    # Split the data into training and test sets
    train_length = floor(length * 0.8)
    test_length = ceil(length * 0.2)
    if train_length + test_length != length:
        raise Exception("You have int problems with the random split.")
    trainDataset, testDataset = torch.utils.data.random_split(TensorDataset(ff_inputs, feat_labels), [train_length, test_length])

    return trainDataset, testDataset


def confirmOrder(label1, label2, label3):
    for l1, l2, l3 in zip(label1, label2, label3):
        if l1.item() != l2.item() and l2.item() != l3.item():
            return False
    return True


def model(trainDataset, testDataset, learning_rate: float, batch_num: int, epoch_num: int, verbose: bool):
    loader = DataLoader(trainDataset, batch_size=batch_num)
    net = FeedForwardNeuralNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    acc = []
    total_loss = []
    for epoch in tqdm(range(0, epoch_num)):  # loop over the dataset multiple times
        loss_in_epoch = []
        for i, data in enumerate(loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if verbose:
                loss_in_epoch.append(loss.item())
        if verbose:
            acc.append(model_test(testDataset, net))
            total_loss.append(mean(loss_in_epoch))
    return acc, total_loss


def model_test(testDataset, net):
    length = len(testDataset)
    loader = DataLoader(testDataset, length)
    test_inputs, test_labels = next(iter(loader))
    test_outputs = net(test_inputs)

    # Convert from onehot to integer
    labels = torch.argmax(test_labels, dim=1)
    predictions = torch.argmax(test_outputs, dim=1)

    # Calculate the accuracy
    acc_test = 0
    for predict, label in zip(predictions, labels):
        if predict == label:
            acc_test += 1
    acc_test = 100 * acc_test / length
    return acc_test


# DNN model
class FeedForwardNeuralNet(nn.Module):
    def __init__(self):
        super(FeedForwardNeuralNet, self).__init__()
        # Initialize the layers
        self.ffLayer1 = nn.Linear(in_features=30, out_features=20, bias=True)
        self.ffLayer2 = nn.Linear(in_features=20, out_features=10, bias=True)

    def forward(self, x):
        """
        :param torch.Tensor x:
        :return:
        """
        x = F.relu(self.ffLayer1(x))
        x = torch.sigmoid(self.ffLayer2(x))
        return x


if __name__ == '__main__':
    trainDataset, testDataset = get_data()
    acc, total_loss = model(trainDataset, testDataset, 0.001, 10, 2125, True)
    fig, axs = plt.subplots(2, sharex='col')
    fig.suptitle('Vertically stacked subplots')
    fig.suptitle('Percent Accuracy and Loss over Epochs')
    axs[0].plot(acc)
    axs[0].set(ylabel='Percent Correct')
    axs[1].plot(total_loss)
    axs[1].set(xlabel='Epoch', ylabel='CrossEntropyLoss')
    plt.show()
