import pickle

import numpy as np

# import scipy
#import rcnn_utils as U
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from main.util import PngData
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from statistics import mean
torch.manual_seed(0)


def train(optimizer, net):
    png_data = pickle.load(open('png_data.pkl', 'rb'))
    train_data = int(len(png_data)* .8)#/3

    test_data = len(png_data) - train_data

    train_set, test_set = random_split(png_data, [int(train_data), int(test_data)])

    dataloader = DataLoader(train_set, batch_size=20)
    #print(train_set.shape())
    #"""

    cel = nn.BCELoss()
    total_loss = []
    epoch_num = 100
    #model = net

    for epoch in range(0, epoch_num + 1):  # loop over the dataset multiple times
        loss_in_epoch = []
        for i, (images, labels) in enumerate(dataloader, 0):
            # get the inputs
            images = Variable(images)
            labels = Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(images)
            loss = cel(outputs, labels)
            # forward + backward + optimize
            loss.backward()
            optimizer.step()
            loss_in_epoch.append(loss.item())
        total_loss.append(mean(loss_in_epoch))

        accuracy = test(net, test_set)
        print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %f %%' % (accuracy))
    print(total_loss)
    print('Finished Training')
    return total_loss



def test(net, test_set):

    correct_count = 0
    count = 0
    x_test, y_test = next(iter(DataLoader(test_set, batch_size=20)))
    pred_test = net(x_test)

    for i in range(len(pred_test)):
        pred_label = torch.argmax(pred_test[i])
        true_label = torch.argmax(y_test[i])
        if pred_label.item() == true_label.item():
            correct_count += 1
        count += 1
    return correct_count / count


class RcnnNet(nn.Module):  # have to change numbers depending on data
    def __init__(self):
        super(RcnnNet, self).__init__()

        self.conv1 = nn.Conv2d(4, 10, 1)
        self.pool = nn.AdaptiveMaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 10, 1)
        self.conv3 = nn.Conv2d(10, 10, 1)
        self.conv4 = nn.Conv2d(10, 64, 1)
        #self.pool = nn.AdaptiveMaxPool2d(2, 2)

        self.fc_1 = nn.Linear(256, 120)
        self.fc_2 = nn.Linear(120, 64)
        self.fc_3 = nn.Linear(64, 10)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x_pre_input = x
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # residual connection
        x = x + x_pre_input
        x = F.relu(self.pool(self.conv4(x)))

        # flattens tensor
        x = x.view(x.size(0), -1)  # number of samples in batch
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = torch.sigmoid(self.fc_3(x))

        return x


if __name__ == '__main__':
    rcnn_0 = RcnnNet()#.to('Gpu')
    #optimizer =
    # optimizer = optim.Adam(rcnn_0.parameters(), lr=0.01)
    result_1 = train(optim.SGD(rcnn_0.parameters(), lr=0.001, weight_decay=1), rcnn_0)
    result_2 = train(optim.Adam(rcnn_0.parameters(), lr=0.001, weight_decay=1), rcnn_0)
    result_3 = train(optim.Adagrad(rcnn_0.parameters(), lr=0.001, weight_decay=1), rcnn_0)
    plt.plot(result_1, 'g', label='SGD')
    plt.plot(result_2, 'b', label='Adam')
    plt.plot(result_3, 'r', label='ADA')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()










