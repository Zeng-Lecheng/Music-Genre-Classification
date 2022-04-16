import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm

from util import WavData

device = 'cpu'

# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()

# torch.set_num_threads(1)


class RNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_1 = nn.LSTM(1, 128, 1, batch_first=True)
        self.lstm_2 = nn.LSTM(128, 32, 1, batch_first=True)
        self.lstm_3 = nn.LSTM(32, 32, 1, batch_first=True)
        self.rnn_1 = nn.RNN(1, 128, 1, batch_first=True)
        self.drop_1 = nn.Dropout(0.5)
        self.drop_2 = nn.Dropout(0.3)
        self.fc_1 = nn.Linear(128, 32)
        self.fc_2 = nn.Linear(32, 12)
        self.fc_3 = nn.Linear(12, 10)

    def forward(self, x):
        # ref: https://www.diva-portal.org/smash/get/diva2:1354738/FULLTEXT01.pdf
        lstm_out, hc = self.rnn_1(x)
        # lstm_out, hc = self.lstm_2(self.drop_1(lstm_out))
        # lstm_out, hc = self.lstm_3(self.drop_2(lstm_out))
        # h_0 = torch.relu(hc[0][0])

        ave_out = torch.sum(lstm_out, dim=1) / lstm_out.shape[1]
        # x = torch.relu(self.fc_1(h_0))
        x = torch.relu(self.fc_1(ave_out))
        x = torch.relu(self.fc_2(x))
        x = torch.sigmoid(self.fc_3(x))
        return x


def model_train(epochs: int,
                learning_rate: float,
                batch_size: int,
                verbose: bool = False,
                test_while_train: bool = True) -> list[float]:
    dataset = WavData('../data/genres_original', device=device)
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    net = RNNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
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
            epoch_loss += batch_loss.item() / len(x_train)

        if test_while_train:
            epoch_acc = model_test(train_set, net)
            acc.append(epoch_acc)
            writer.add_scalar('Accuracy/test', epoch_acc, epoch)
            if verbose:
                print(f'Epoch: {epoch} Loss: {epoch_loss} Accuracy: {epoch_acc}')
        elif verbose:
            print(f'Epoch: {epoch} Loss: {epoch_loss}')

        writer.add_scalar('Loss/train', epoch_loss, epoch)

    return acc


def model_test(test_set, net) -> float:
    with torch.no_grad():
        correct_count = 0
        count = 0
        x_test, y_test = next(iter(DataLoader(test_set, batch_size=200, shuffle=True)))
        pred_test = net(x_test)

        for i in range(len(pred_test)):
            pred_label = torch.argmax(pred_test[i])
            true_label = torch.argmax(y_test[i])
            if pred_label.item() == true_label.item():
                correct_count += 1
            count += 1
    return correct_count / count


if __name__ == '__main__':
    acc = model_train(epochs=5000, learning_rate=.0005, batch_size=50, verbose=False, test_while_train=True)
    plt.plot(range(1, len(acc) + 1), acc)
    plt.show()
