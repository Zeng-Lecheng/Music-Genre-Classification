import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime

from util import WavData

from torch.utils.tensorboard import SummaryWriter

# fixed : pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# use cpu by default, change to cuda if you want and change it back before committing
# we only debug and ensure everything works well on cpu
device = 'cuda'

# uncomment to run with limited cores
# torch.set_num_threads(1)
torch.manual_seed(0)


class RNNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm_1 = nn.LSTM(1, 16, 1, batch_first=True)
        self.lstm_2 = nn.LSTM(32, 12, 1, batch_first=True)
        self.lstm_3 = nn.LSTM(32, 32, 1, batch_first=True)
        self.rnn_1 = nn.RNN(1, 64, 1, batch_first=True)
        self.rnn_2 = nn.RNN(64, 32, 1, batch_first=True)
        self.rnn_3 = nn.RNN(32, 32, 1, batch_first=True)
        self.rnn_4 = nn.RNN(32, 32, 1, batch_first=True)
        self.drop_1 = nn.Dropout(0.5)
        self.drop_2 = nn.Dropout(0.3)
        self.fc_1 = nn.Linear(32, 20)
        self.fc_1_conv = nn.Linear(1776, 20)
        self.fc_2 = nn.Linear(20, 16)
        self.fc_3 = nn.Linear(16, 10)

        self.conv_1 = nn.Conv1d(32, 16, 5)
        self.conv_2 = nn.Conv1d(16, 8, 5)

    def forward(self, x):
        # ref: https://www.diva-portal.org/smash/get/diva2:1354738/FULLTEXT01.pdf
        x, hc = self.lstm_1(x)
        # x, hc = self.lstm_2(x)
        # out, hc = self.lstm_2(self.drop_1(out))
        # out, hc = self.lstm_3(self.drop_2(out))
        # h_0 = torch.relu(hc[0][0])
        hidden_state = torch.relu(hc[0][0])

        # ave_out = torch.sum(x, dim=1) / x.shape[1]
        # x = torch.swapaxes(x, 1, 2)
        # x = torch.relu(torch.max_pool1d(self.conv_1(x), 2))
        # x = torch.relu(torch.max_pool1d(self.conv_2(x), 2))
        # x = torch.flatten(hidden_state, 1)
        # x = torch.relu(self.fc_1_conv(x))
        # x = torch.relu(self.fc_1(hidden_state))
        # x = torch.relu(self.fc_2(x))
        x = torch.sigmoid(self.fc_3(hidden_state))
        return x


def model_train(train_set, test_set,
                epochs: int,
                learning_rate: float,
                batch_size: int,
                weight_decay: float,
                verbose: bool = False,
                test_while_train: bool = True) -> list[float]:

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    net = RNNet().to(device)
    # net.load_state_dict(torch.load('../saved_models/rnn_with_cov_final.pt'))
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)

    acc = []
    for epoch in tqdm(range(1, epochs + 1)):
        epoch_loss = 0
        for x_train, y_train in train_dataloader:
            optimizer.zero_grad()
            pred_train = net(x_train)
            batch_loss = criterion(pred_train, y_train)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item() * len(x_train)

        epoch_loss = epoch_loss / len(train_set)
        if test_while_train:
            test_acc = model_test(test_set, net)
            train_acc = model_test(train_set, net)
            acc.append(test_acc)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            if verbose:
                print(f'Epoch: {epoch} Loss: {epoch_loss} Accuracy: {test_acc}')
        elif verbose:
            print(f'Epoch: {epoch} Loss: {epoch_loss}')

        writer.add_scalar('Loss/train', epoch_loss, epoch)

    # Uncomment this if you want to save the trained model.
    # torch.save(net.state_dict(), f'../saved_models/rnn_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
    # torch.save(net.state_dict(), '../saved_models/rnn_with_cov_final.pt')
    if not test_while_train:
        acc = [model_test(test_set, net)]
    return acc


def model_test(test_set, net) -> float:
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


def get_data():
    dataset = WavData('../data/genres_original', device=device)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    return train_set, test_set


def hyperparameter_test():
    train_set, test_set = get_data()
    learning_rate_list = [0.001, 0.0005, 0.0003, 0.0001, 0.00005]
    batch_size_list = [50, 100, 200]
    epochs = 3
    weight_decay = 0.002
    for b in batch_size_list:
        for lr in learning_rate_list:
            acc = model_train(train_set, test_set, epochs=epochs, learning_rate=lr, weight_decay=weight_decay, batch_size=b, verbose=False,
                              test_while_train=True)
            plt.plot(range(1, epochs + 1), acc, label=f'batches: {b}, lr: {lr}')

        plt.xlabel("Epoch")
        plt.ylabel("Percent Accuracy on test set")
        plt.title('Percent Accuracy over Epochs')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    epochs = 1000
    lr = .0005
    batch = 60
    weight_decay = 0
    train_set, test_set = get_data()
    writer = SummaryWriter(comment=f'lr{lr}_batch{batch}_l2{weight_decay}')
    acc = model_train(train_set, test_set, epochs=epochs, learning_rate=lr, batch_size=batch, weight_decay=weight_decay, verbose=False,
                      test_while_train=True)
