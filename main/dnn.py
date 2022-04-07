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


def main():
    x, y, length = util.loadCSV(f'../data/features_30_sec.csv')
    train_length = int(length * 0.8)
    test_length = int(length * 0.2)
    if train_length + test_length != length:
        raise Exception("you have int problems with the random split")
    trainDataset, testDataset = torch.utils.data.random_split(TensorDataset(x, y), [train_length, test_length])

    result_list = {}
    lr_list = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    b_list = [5, 10, 50, 100, 500, 1000]
    e_list = [100, 100, 1000, 2000, 5000]
    for lr in lr_list:
        for b in b_list:
            for e in e_list:
                key = "Model Run lr=", lr, " b=", b, " e=", e
                print(key)
                result = model(trainDataset, testDataset, lr, b, e)
                print(result)
                result_list[key] = result


def model(trainDataset, testDataset, learning_rate: float, batch_num: int, epoch_num: int):
    loader = DataLoader(trainDataset, batch_size=batch_num)
    epochs = epoch_num
    net = DeepNeuralNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    print("Progress: ", end="")
    for epoch in range(0, epochs):  # loop over the dataset multiple times
        for i, data in enumerate(loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print("#", end="")
    print()
    return model_test(testDataset, net)


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
        x = F.sigmoid(self.ffLayer2(x))
        return x


main()
