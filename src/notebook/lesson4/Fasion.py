import torch
from torch import nn, optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

import src.notebook.assets.h.helper


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, 10)
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Now with dropout
        x = self.dropout(F.relu(self.layer1(x)))
        x = self.dropout(F.relu(self.layer2(x)))
        x = self.dropout(F.relu(self.layer3(x)))
        x = self.dropout(F.relu(self.layer4(x)))

        # output so no dropout here
        x = F.log_softmax(self.layer5(x), dim=1)

        return x


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_set = datasets.mnist.FashionMNIST('~/.pytorch/F_MNIST_data/',
                                        download=True,
                                        train=True, transform=transform)

test_set = datasets.mnist.FashionMNIST('~/.pytorch/F_MNIST_data/',
                                       download=True,
                                       train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

model = Network()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 5
steps = 0
train_loss, test_loss = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    else:
        testing_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in test_loader:
                # no need to optimizer zero gradient
                test_ps = model(images)
                t_loss = criterion(test_ps, labels)
                testing_loss += t_loss

                # probability
                ps = torch.exp(test_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class = labels.view(*top_class.shape)
                mean = torch.mean(equals.type(torch.FloatTensor))
                accuracy += mean
        # back to train
        model.train()
        train_loss.append(running_loss / len(train_loader))
        test_loss.append(testing_loss / len(test_loader))

    print("Epoch: {}/{}.. ".format(e + 1, epochs),
          "Training Loss: {:.3f}.. ".format(train_loss[-1]),
          "Test Loss: {:.3f}.. ".format(test_loss[-1]),
          "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)))

    print("Epoch:{}/{}..".format(e + 1, epochs),
          "Training Loss: {:.3f}..".format(train_loss[-1]),
          "Test Loss: {:.3f}..".format(test_loss[-1]),
          "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)))
