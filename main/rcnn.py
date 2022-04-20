import numpy as np
import matplotlib.pyplot as plt
# import scipy
#import rcnn_utils as U
import torch
from torch.utils.data import DataLoader, random_split
from main.util import PngData
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from statistics import mean
torch.manual_seed(0)


def train():
    train_data = int(len(PngData('../small_data/images_original'))* .7)#/3
    print(train_data)
    test_data = len(PngData('../small_data/images_original')) - train_data
    print(test_data)
    train_set, test_set = random_split(PngData('../small_data/images_original'), [int(train_data), int(test_data)])
    print(len(train_set))
    print(len(test_set))
    dataloader = DataLoader(train_set, batch_size=10)
    #print(train_set.shape())
    #"""

    cel = nn.CrossEntropyLoss()
    total_loss = []
    epoch_num = 10#100
    rcnn_0 = RcnnNet().to('cpu')
    optimizer = optim.Adam(rcnn_0.parameters(), lr=0.0001)
    for epoch in range(0, epoch_num + 1):  # loop over the dataset multiple times
        loss_in_epoch = []
        for i, (images, labels) in enumerate(dataloader, 0):
            #print(i)
            # get the inputs
            images = Variable(images.to('cpu'))
            labels = Variable(labels.to('cpu'))
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = rcnn_0(images)
            loss = cel(torch.squeeze(outputs), torch.squeeze(labels))
            # forward + backward + optimize
            loss.backward()
            optimizer.step()
            loss_in_epoch.append(loss.item())
        total_loss.append(mean(loss_in_epoch))

        #accuracy = test(rcnn_0, test_set)
        print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %f %%' % (accuracy))
    print(total_loss)
    print('Finished Training')




def test(net, test_set):
    correct_count = 0
    count = 0
    x_test, y_test = next(iter(DataLoader(test_set, batch_size=10)))
    pred_test = net(x_test)

    for i in range(len(pred_test)):
        pred_label = torch.argmax(pred_test[i])
        true_label = torch.argmax(y_test[i])
        if pred_label.item() == true_label.item():
            correct_count += 1
        count += 1
    return correct_count / count


"""


    net.eval()
    accuracy = 0.0
    total = 0.0
    dataloader = DataLoader(test_set, batch_size=5)
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader, 0):
            print(i)
            # get the inputs
            images = Variable(images.to('cpu'))
            labels = Variable(labels.to('cpu'))
            # run the model on the test set to predict labels
            outputs = net(images)  #
            # highest label represents the predictions
            _, prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (prediction == labels).sum().item()
                    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return accuracy

"""


class RcnnNet(nn.Module):  # have to change numbers depending on data
    def __init__(self):
        super(RcnnNet, self).__init__()

        self.conv1 = nn.Conv2d(4, 10, 3)
        self.pool = nn.AdaptiveMaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.pool = nn.AdaptiveMaxPool2d(2, 2)

        self.fc_1 = nn.Linear(64 * 56 * 56, 120)
        self.fc_2 = nn.Linear(120, 64)
        self.fc_3 = nn.Linear(64, 2)
        self.Sigmoid = nn.Sigmoid()
        """
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.pool = nn.AdaptiveMaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)

        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.pool = nn.AdaptiveMaxPool2d(2, 2)

        self.fc_1 = nn.Linear(64 * 56 * 56, 120)
        self.fc_2 = nn.Linear(120, 64)
        self.fc_3 = nn.Linear(64, 2)
        self.Sigmoid = nn.Sigmoid()
        """


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x_pre_input = x
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # residual connection
        x = x + x_pre_input
        x = F.relu(self.conv4(x))

        # flattens tensor
        x = x.view(x.size(0), -1)  # number of samples in batch
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.sigmoid(self.fc_3(x))
        """
         x_pre_input = x
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # residual connection
        x = x + x_pre_input
        x = F.relu(self.conv4(x))

        # flattens tensor
        x = x.view(x.size(0), -1)  # number of samples in batch
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.sigmoid(self.fc_3(x))
        """

        return x


train()


