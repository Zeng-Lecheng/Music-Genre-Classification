import numpy as np
import matplotlib.pyplot as plt
# import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import util

torch.manual_seed(0)


def main(seconds: str):
    # Loading the data
    x, y, length = util.loadCSV(f'../data/features_{seconds}_sec.csv')
    train_length = int(length * 0.8)
    test_length = int(length * 0.2)
    if train_length + test_length != length:
        raise Exception("you have int problems with the random split")
    trainDataset, testDataset = torch.utils.data.random_split(TensorDataset(x, y), [train_length, test_length])
    batches = 10
    loader = DataLoader(trainDataset, batch_size=batches)

    # Setting up the neural net and components to make it run
    epochs = 1
    net = DeepNeuralNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    acc = []
    print("Progress:")
    for epoch in range(0, epochs):  # loop over the dataset multiple times
        for i, data in enumerate(loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        acc.append(model_test(testDataset, net, test_length))
        print("#", end="")
    print()
    print(acc)


def model_test(testDataset, net, length):
    loader = DataLoader(testDataset, batch_size=length)
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


main(str(30))
