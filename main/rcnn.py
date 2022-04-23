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

def data_loader():
    png_data = pickle.load(open('png_data.pkl', 'rb'))
    train_data = int(len(png_data) * .8)  # /3

    test_data = len(png_data) - train_data

    train_set, test_set = random_split(png_data, [int(train_data), int(test_data)])
    return train_set, test_set

def train(optimizer, net, train_set, test_set):
    net.train()
    dataloader = DataLoader(train_set, batch_size=20, shuffle = True)
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
    net.eval()
    batch_size = 20
    x_test, y_test = next(iter(DataLoader(test_set, batch_size=20)))
    #pred_test = net(x_test)
    count = 0.0
    total = 0.0

    with torch.no_grad():
        for j in range(0, batch_size):
            inputs = x_test
            labels = y_test
            # run the model on the test set to predict labels
            outputs = net(inputs)  #
            # highest label represents the predictions
            _, prediction = torch.max(outputs.data, 1)
            _, actual = torch.max(labels.data, 1)
            total += labels.size(0)
            for i in range(0, len(labels)):
                if actual[i] == prediction[i]:
                    count += 1
                    # compute the accuracy over all test images
    accuracy = (100 * count / total)
    return accuracy

    """
    for i in range(len(pred_test)):
        pred_label = torch.argmax(pred_test[i])
        true_label = torch.argmax(y_test[i])
        if pred_label.item() == true_label.item():
            correct_count += 1
        count += 1
    return correct_count / count
"""

class RcnnNet(nn.Module):  # have to change numbers depending on data
    def __init__(self):
        super(RcnnNet, self).__init__()

        self.conv1 = nn.Conv2d(4, 8, 3, 1)
        self.pool = nn.AdaptiveMaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.conv3 = nn.Conv2d(8, 8, 3, 1)
        self.conv4 = nn.Conv2d(8, 16, 3, 1)
        #self.pool = nn.AdaptiveMaxPool2d(2, 2)

        self.fc_1 = nn.Linear(64, 120)
        self.fc_2 = nn.Linear(120, 64)
        self.dropout = nn.dropout(.5)
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
        x = self.dropout(x)
        x = F.relu(self.fc_2(x))
        x = torch.sigmoid(self.fc_3(x))

        return x


if __name__ == '__main__':
    rcnn_0 = RcnnNet()#.to('Gpu')
    train_set, test_set = data_loader()
    result_1 = train(optim.SGD(rcnn_0.parameters(), lr=0.001, weight_decay=.01), rcnn_0, train_set, test_set)
    result_2 = train(optim.Adam(rcnn_0.parameters(), lr=0.001, weight_decay=.01), rcnn_0, train_set, test_set)
    result_3 = train(optim.Adagrad(rcnn_0.parameters(), lr=0.001, weight_decay=.01), rcnn_0, train_set, test_set)
    plt.plot(result_1, 'g', label='SGD')
    plt.plot(result_2, 'b', label='Adam')
    plt.plot(result_3, 'r', label='ADA')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()










