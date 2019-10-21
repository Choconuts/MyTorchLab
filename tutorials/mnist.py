import torch
import numpy as np
import math
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn

from torchvision import datasets
from torchvision.transforms import transforms

BATCH_SIZE = 80
LEARNING_RATE = 1e-3
EPOCH = 100
SHOW_STEPS = 1000
CUDA = torch.cuda.is_available()


data_loader = DataLoader(
    datasets.MNIST(
        root='data/mnist',
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.full = nn.Sequential(
            nn.Linear(256, 120),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(120, 84),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(84, 10),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor):
        img = img.view(img.size(0), 1, 28, 28)
        features = self.conv(img)
        features = features.view(img.size(0), -1)
        print(features.shape)
        logits = self.full(features)
        return logits


loss_func = torch.nn.CrossEntropyLoss()


def evaluate(logits, label):
    return torch.sum(torch.argmax(logits, 1, True).eq(label.view(-1, 1))) * 1. / logits.size(0)


classifier = Classifier()

if CUDA:
    print('cuda is available')
    classifier.cuda()
    loss_func.cuda()

optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)


for epoch in range(EPOCH):
    for i, (images, label) in enumerate(data_loader):
        assert isinstance(images, torch.Tensor) and isinstance(label, torch.Tensor)

        optimizer.zero_grad()
        logits = classifier(images.float().cuda())
        loss = loss_func(logits, label.long().cuda())
        loss.backward()
        optimizer.step()
        if i % SHOW_STEPS == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [accu: %f]"
                % (epoch, 1000, i, len(data_loader), loss.item(), evaluate(logits, label.cuda()))
            )


if __name__ == '__main__':
    pass