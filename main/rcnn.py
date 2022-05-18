import pickle

import numpy as np

# import scipy
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from util import PngData
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)

# fixed : pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# use cpu by default, change to cuda if you want and change it back before committing
# we only debug and ensure everything works well on cpu
device = 'cuda'


def data_loader():
    png_data = PngData('../data/images_original', device=device)
    train_data = int(len(png_data) * .8)  # /3

    test_data = len(png_data) - train_data

    train_set, test_set = random_split(png_data, [int(train_data), int(test_data)])
    return train_set, test_set


def model_train(optimizer, net, epochs, batch_size, train_set, test_set, verbose=False):
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # net.load_state_dict(torch.load('../saved_models/rnn_with_cov_final.pt'))
    cel = nn.CrossEntropyLoss().to(device)
    total_loss = []
    epoch_num = epochs

    for epoch in tqdm(range(0, epoch_num)):  # loop over the dataset multiple times
        epoch_loss = 0
        net.train()
        for images, labels in dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(images)
            loss = cel(outputs, labels)
            # forward + backward + optimize
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(images)

        epoch_loss = epoch_loss / len(train_set)
        total_loss.append(epoch_loss)
        test_acc = model_test(net, test_set)
        train_acc = model_test(net, train_set)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        if verbose:
            print(f'Epoch: {epoch}, Test accuracy: {test_acc}, Loss: {epoch_loss}')
    # torch.save(net.state_dict(), '../saved_models/rcn.pt')
    return total_loss


def model_test(net, test_set):
    net.eval()
    with torch.no_grad():
        correct_count = 0
        count = 0
        x_test, y_test = next(iter(DataLoader(test_set, batch_size=20, shuffle=True)))
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

        self.conv1 = nn.Conv2d(4, 8, 5, 1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 8, 1, 1)
        self.conv3 = nn.Conv2d(8, 8, 1, 1)
        self.conv4 = nn.Conv2d(8, 16, 5, 1)
        self.conv5 = nn.Conv2d(16, 32, 5, 1)
        self.conv6 = nn.Conv2d(32, 32, 1, 1)
        self.conv7 = nn.Conv2d(32, 32, 1, 1)
        self.conv8 = nn.Conv2d(32, 64, 5, 1)
        self.conv9 = nn.Conv2d(64, 128, 5, 1)
        self.conv10 = nn.Conv2d(128, 256, 5, 1)

        self.fc_1 = nn.Linear(20608, 120)
        self.fc_2 = nn.Linear(120, 64)
        self.dropout = nn.Dropout(.5)
        self.fc_3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.pool(self.conv1(x)))
        # x_pre_input = x
        # x = torch.relu(self.conv2(x))
        # x = torch.relu(self.conv3(x))
        # # residual connection
        # x = x + x_pre_input
        x = torch.relu(self.pool(self.conv4(x)))
        x = torch.relu(self.pool(self.conv5(x)))
        # x_pre_input = x
        # x = torch.relu(self.conv6(x))
        # x = torch.relu(self.conv7(x))
        # x = x + x_pre_input
        x = torch.relu(self.pool(self.conv8(x)))
        # x = torch.relu(self.pool(self.conv9(x)))
        # x = torch.relu(self.pool(self.conv10(x)))

        # flattens tensor
        x = x.view(x.size(0), -1)  # number of samples in batch
        x = torch.relu(self.fc_1(x))
        x = torch.relu(self.fc_2(x))
        x = torch.sigmoid(self.fc_3(x))

        return x


if __name__ == '__main__':
    writer = SummaryWriter(comment='cnn2_lr0.0005_batch20_wd0.0001')
    rcnn_0 = RcnnNet().to(device)
    train, test = data_loader()
    adam = optim.Adam(rcnn_0.parameters(), lr=0.0005, weight_decay=.0001)
    result = model_train(adam, rcnn_0, epochs=1000, batch_size=20, train_set=train, test_set=test)
