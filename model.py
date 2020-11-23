## 딥러닝 모델
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(in_features=118, out_features=1000)
        self.bnorm1 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=1000)
        self.bnorm2 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(in_features=1000, out_features=100)
        self.bnorm3 = nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(in_features=100, out_features=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bnorm1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bnorm2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.bnorm3(x)
        x = self.relu(x)

        x = self.fc4(x)
        x = self.sigmoid(x)

        return x

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.fc1 = nn.Linear(in_features=118, out_features=100)
        self.bnorm1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(in_features=100, out_features=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bnorm1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bnorm2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.bnorm3(x)
        x = self.relu(x)

        x = self.fc4(x)
        # x = self.sigmoid(x)

        return x