import numpy as npimport matplotlib.pyplot as plt# import scipy#import rcnn_utils as Uimport torch"""import torch.nn as nnfrom torch.autograd import Variableimport torch.nn.functional as Fimport torch.optim as optimfrom torch.utils.data import TensorDataset, DataLoaderfrom statistics import mean"""torch.manual_seed(0)from torch.utils.data import DataLoader#from main.util import WavDatafrom main.util import PngDatadataloader = DataLoader(PngData('../data/images_original'))for x, y in dataloader:    print(x.shape)    print(y.shape)"""def main():    train(train_data_x, train_data_y, test_data_x, test_data_y, optimizer):    def train(train_data_x, train_data_y, test_data_x, test_data_y, optimizer):     rnn_0 = RCNN_Net.train()     batch_size = 5      cel = nn.CrossEntropyLoss()     total_loss = []     epoch_num = 100     for epoch in range(epoch_num):  # loop over the dataset multiple times         loss_in_epoch = []         for i in range(0, batch_size):             inputs = Variable(train_data_x)             labels = Variable(train_data_y)             # zero the parameter gradients             outputs = rnn_0(inputs)             loss = cel(torch.squeeze(outputs), torch.squeeze(labels))               optimizer.zero_grad()             # forward + backward + optimize             loss.backward()             optimizer.step()             loss_in_epoch.append(loss.item())         total_loss.append(mean(loss_in_epoch))              accuracy =  test(test_data_x, test_data_y, rnn_0, batch_size)         print('For epoch', epoch+1,'the test accuracy over the whole test set is %f %%' % (accuracy))               print('Finished Training')def test(test_data_x, test_data_y, net, batch_size):        net.eval()        count = 0.0        total = 0.0        with torch.no_grad():            for i in range(0, batch_size):                inputs = test_data_x                labels = test_data_y                # run the model on the test set to predict labels                outputs = net(inputs) #                #highest label represents the predictions                 _, prediction = torch.max(outputs.data, 1)                total += labels.size(0)                for i in range (0, len(labels)):                    if labels[i] == prediction[i]:                        count += 1         # compute the accuracy over all test images        accuracy = (100 * count / total)        return accuracyclass RCNN_Net(nn.Module):  # have to change numbers depending on data    def __init__(self):      super(Net, self).__init__()                 self.conv1 = nn.Conv2d(3, 6, 5, 1)      self.pool = nn.AdaptiveMaxPool2d(2,2)      self.conv2 = nn.Conv2d(6, 6, 1, 1)            self.conv3 = nn.Conv2d(6, 6, 1, 1)      self.conv4 = nn.Conv2d(6, 12, 5, 1)      self.pool = nn.AdaptiveMaxPool2d(2,2)            self.fc_1 = nn.Linear(12 * 56 * 56, 120)      self.fc_2 = nn.Linear(120, 64)      self.fc_3 = nn.Linear(64, 2)      self.Sigmoid = nn.Sigmoid()   def forward(self, x):       x = F.relu(self.conv1(x))              x_pre_input = x       x = F.relu(self.conv2(x))        x = F.relu(self.conv3(x))       #residual connection       x = x + x_pre_input       x = F.relu(self.conv4(x))              #flattens tensor       x = x.view(x.size(0), -1) #number of samples in batch       x = F.relu(self.fc_1(x))       x = F.relu(self.fc_2(x))       x = F.sigmoid(self.fc_3(x))                return x """