from statistics import mean
import numpy as np
import math
import matplotlib.pyplot as plt
# import scipy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import util

torch.manual_seed(0)


def hyperparameter_test(trainDataset, testDataset):
    """This function tests out different learning rates and batch sizes and plots them over epochs.
    It becomes clear that LR=0.001 and batch size=10 produces the most consistently best results,
    but is that due to the seed we are currently using? What if I plotted multiple different seeds?"""
    lr_list = [0.0005, 0.001, 0.005, 0.01]
    b_list = [5, 10, 50, 100, 500, 1000]

    for lr in lr_list:
        fig, ax = plt.subplots()
        for b in b_list:
            acc, loss = model(trainDataset, testDataset, lr, b, 5000, True)
            ax.plot(acc, label=str(f'Batches={b}, LR={lr}'))
        ax.set_title("Percent Accuracy over Epochs")
        ax.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Percent Accuracy")
        plt.show()


def read_data():
    """I separated out reading the data from the model so that the model could be run multiple times without having
    to reread the data in every time."""
    x, y, length = util.loadCSV(f'../data/features_30_sec.csv')
    train_length = math.floor(length * 0.8)
    test_length = math.ceil(length * 0.2)
    if train_length + test_length != length:
        raise Exception("you have int problems with the random split")
    trainDataset, testDataset = torch.utils.data.random_split(TensorDataset(x, y), [train_length, test_length])
    return trainDataset, testDataset


def model(trainDataset, testDataset, learning_rate: float, batch_num: int, epoch_num: int, verbose: bool):
    loader = DataLoader(trainDataset, batch_size=batch_num)
    net = DeepNeuralNet()
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
    #torch.save(net.state_dict(), '../saved_models/dnn.pt')
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
class DeepNeuralNet(nn.Module):
    def __init__(self):
        super(DeepNeuralNet, self).__init__()
        # Initialize the layers
        self.ffLayer1 = nn.Linear(in_features=58, out_features=34, bias=True)
        self.ffLayer2 = nn.Linear(in_features=34, out_features=10, bias=True)

    def forward(self, x):
        """
        :param torch.Tensor x:
        :return:
        """
        x = F.relu(self.ffLayer1(x))
        x = torch.sigmoid(self.ffLayer2(x))
        return x


if __name__ == '__main__':
    trainDataset, testDataset = read_data()
    acc, total_loss = model(trainDataset, testDataset, 0.001, 10, 2125, True)
    fig, axs = plt.subplots(2, sharex='col')
    fig.suptitle('Vertically stacked subplots')
    fig.suptitle('Percent Accuracy and Loss over Epochs')
    axs[0].plot(acc)
    axs[0].set(ylabel='Percent Correct')
    axs[1].plot(total_loss)
    axs[1].set(xlabel='Epoch', ylabel='CrossEntropyLoss')
    plt.show()

