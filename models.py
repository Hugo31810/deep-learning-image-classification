import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()

        # Primer bloque convolucional
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Segundo bloque convolucional
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        # Tercer bloque convolucional
        self.conv3 = nn.Conv2d(16, 20, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(20)

        # Capa de aplanado
        self.flatten = nn.Flatten()

        # Linear para la clasificación
        self.fc = nn.Linear(20*7*7, output_dim)

    def forward(self, x):
        # Primer bloque convolucional
        x = F.relu(self.bn1(self.conv1(x)))

        # Segundo bloque convolucional
        x = F.relu(self.bn2(self.conv2(x)))

        # Tercer bloque convolucional
        x = F.relu(self.bn3(self.conv3(x)))

        # Aplanar la salida de la última capa convolucional
        x = self.flatten(x)

        # Linear para clasificación
        x = self.fc(x)

        return x


class FCNN(nn.Module):
    def __init__(self, output_dim):
        super(FCNN, self).__init__()

        # Primer bloque convolucional
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Segundo bloque convolucional
        self.conv2 = nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(24)

        # Tercer bloque convolucional
        self.conv3 = nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Cuarto bloque convolucional
        self.conv4 = nn.Conv2d(32, 38, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(38)

        # Bloque de salida
        self.output_conv = nn.Conv2d(38, output_dim, kernel_size=4)

    def forward(self, x):
        # Primer bloque convolucional
        x = F.relu(self.bn1(self.conv1(x)))

        # Segundo bloque convolucional
        x = F.relu(self.bn2(self.conv2(x)))

        # Tercer bloque convolucional
        x = F.relu(self.bn3(self.conv3(x)))

        # Cuarto bloque convolucional
        x = F.relu(self.bn4(self.conv4(x)))

        # Capa de salida
        x = self.output_conv(x) # Salida (N, num_classes, 1, 1)
        x = x.view(x.size(0), -1)  # (N, num_classes)
        return x
