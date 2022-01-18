import torch
import torch.nn as nn


class Tnet(nn.Module):
    ''' the transformation network preceding point net'''

    def __init__(self, k=3):
        super(Tnet, self).__init__()

        self.k = k
        self.conv1 = nn.Conv1d(
            in_channels=self.k, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=1024, kernel_size=1)

        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=self.k*self.k)

        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.bn3 = nn.BatchNorm1d(num_features=1024)
        self.bn4 = nn.BatchNorm1d(num_features=512)
        self.bn5 = nn.BatchNorm1d(num_features=256)

    def forward(self, input):
        # input.shape == (bs, n, 3)
        batch_size = input.size(0)

        xb1 = nn.functional.relu(self.bn1(self.conv1(input)))
        xb2 = nn.functional.relu(self.bn2(self.conv2(xb1)))
        xb3 = nn.functional.relu(self.bn3(self.conv3(xb2)))
        pool = nn.MaxPool1d(xb3.size(-1))(xb3)
        flat = nn.Flatten(1)(pool)

        xb4 = nn.functional.relu(self.bn4(self.fc1(flat)))
        xb5 = nn.functional.relu(self.bn5(self.fc2(xb4)))

        # initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(batch_size, 1, 1)
        if xb5.is_cuda:
            init = init.cuda()
        matrix = self.fc3(xb5).view(-1, self.k, self.k) + init
        return matrix


class Transform(nn.Module):
    ''' input and feature transformations'''

    def __init__(self):
        super(Transform, self).__init__()

        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)

        self.conv1 = nn.Conv1d(
            in_channels=3, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=1024, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.bn3 = nn.BatchNorm1d(num_features=1024)

    def forward(self, input):
        # input.shape == (bs, n, 3)
        batch_size = input.size(0)

        # batch matrix multiplication
        matrix3x3 = self.input_transform(input)
        xb1 = torch.bmm(torch.transpose(input, 1, 2),
                        matrix3x3).transpose(1, 2)
        xb2 = nn.functional.relu(self.bn1(self.conv1(xb1)))

        matrix64x64 = self.feature_transform(xb2)
        xb3 = torch.bmm(torch.transpose(xb2, 1, 2),
                        matrix64x64).transpose(1, 2)

        xb4 = nn.functional.relu(self.bn2(self.conv2(xb3)))
        xb5 = nn.functional.relu(self.bn3(self.conv3(xb4)))

        xb6 = nn.MaxPool1d(xb5.size(-1))(xb5)
        output = nn.Flatten(1)(xb6)

        return output, matrix3x3, matrix64x64


class PointNet(nn.Module):
    ''' the pointnet network'''

    def __init__(self, classes=10):
        super(PointNet, self).__init__()

        self.transform = Transform()

        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=classes)

        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb1, matrix3x3, matrix64x64 = self.transform(input)
        xb2 = nn.functional.relu(self.bn1(self.fc1(xb1)))
        xb3 = nn.functional.relu(self.bn2(self.dropout(self.fc2(xb2))))
        output = self.fc3(xb3)
        return self.logsoftmax(output), matrix3x3, matrix64x64
