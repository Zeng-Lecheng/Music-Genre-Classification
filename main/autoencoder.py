import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from util import PngData, model_test
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


device = 'cuda'


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(4, 16, 5)
        self.conv_2 = nn.Conv2d(16, 8, 5)
        self.conv_3 = nn.Conv2d(8, 2, 5)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = torch.relu(self.pool(self.conv_1(x)))
        x = torch.relu(self.pool(self.conv_2(x)))
        x = torch.relu(self.pool(self.conv_3(x)))
        return torch.sigmoid(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv_1 = nn.ConvTranspose2d(2, 8, 5)
        self.deconv_2 = nn.ConvTranspose2d(8, 16, 5)
        self.deconv_3 = nn.ConvTranspose2d(16, 4, 5)
        self.upsample_1 = nn.Upsample((65, 101))
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = torch.relu(self.deconv_1(self.upsample_1(x)))
        x = torch.relu(self.deconv_2(self.upsample(x)))
        x = torch.relu(self.deconv_3(self.upsample(x)))

        return x


def perceptual_loss():
    pass


def model_train(train_set, epochs: int, batch: int, lr: float, verbose=False):
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    loss = nn.MSELoss().to(device)
    encoder_optim = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optim = optim.Adam(decoder.parameters(), lr=lr)

    train_dataloader = DataLoader(train_set, batch_size=batch)

    for i in tqdm(range(epochs)):
        epoch_loss = 0
        for batch_x, batch_y in train_dataloader:
            encoder_optim.zero_grad()
            decoder_optim.zero_grad()

            code = encoder(batch_x)
            output = decoder(code)
            batch_loss = loss(output, batch_x)
            batch_loss.backward()
            encoder_optim.step()
            decoder_optim.step()

            epoch_loss += batch_loss.item() * len(batch_x)

        epoch_loss /= len(train_set)
        writer.add_scalar('Loss/train', epoch_loss, i + 1)

    return encoder, decoder


def get_data():
    dataset = PngData('../data/images_original/', device=device)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    return train_set, test_set


if __name__ == '__main__':
    learning_rate = 0.001
    batch_size = 15
    writer = SummaryWriter(comment=f'lr{learning_rate}_batch{batch_size}')

    train_set, test_set = get_data()

    encoder, decoder = model_train(train_set, 100, batch_size, learning_rate)
    test_img = next(iter(DataLoader(test_set, batch_size=1)))
    test_output = decoder(encoder(test_img))
    save_image(test_img, 'input.png')
    save_image(test_output, 'output.png')
