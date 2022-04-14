import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm

from util import WavData

device = 'cuda'

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

torch.set_num_threads(2)


class RNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 16, 1, batch_first=True)
        self.fc_1 = nn.Linear(16, 16)
        self.fc_2 = nn.Linear(16, 12)
        self.fc_3 = nn.Linear(12, 10)

    def forward(self, x):
        lstm_out, hc = self.lstm(x)
        h_0 = torch.relu(hc[0][0])

        ave_out = torch.relu(torch.sum(lstm_out, dim=1))
        # x = torch.relu(self.fc_1(h_0))
        x = torch.relu(self.fc_1(ave_out))
        x = torch.relu(self.fc_2(x))
        return torch.sigmoid(self.fc_3(x))


def model_train(learning_rate: float, batch_size: int, verbose: bool = False, test_while_train: bool = True) -> list[float]:
    dataset = WavData('../data/genres_original', device=device)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_set, batch_size=batch_size)
    net = RNNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)

    acc = []
    for epoch in tqdm(range(1, 101)):
        epoch_loss = 0
        for x_train, y_train in train_dataloader:
            optimizer.zero_grad()
            pred_train = net(x_train)
            batch_loss = criterion(pred_train, y_train)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item() / len(x_train)

        if test_while_train:
            epoch_acc = model_test(test_set, net)
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
        for x_test, y_test in DataLoader(test_set):
            pred_test = net(x_test)
            pred_label = (pred_test == pred_test.max())[0].nonzero()[0]
            true_label = (y_test == 1)[0].nonzero()[0]
            if pred_label.item() == true_label.item():
                correct_count += 1
            count += 1
    return correct_count / count


if __name__ == '__main__':
    acc = model_train(0.01, 10, True)
    plt.plot(range(1 + len(acc)), acc)
    plt.show()
