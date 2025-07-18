import torch.nn as nn
from torch import Tensor, concat, flatten


class Network(nn.Module):
    def __init__(self, input_size: int, output_layer1: int, output_layer2: int) -> None:
        super().__init__()

        self.linear1 = nn.Linear(input_size, output_layer1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=output_layer1)

        self.linear2 = nn.Linear(output_layer1, output_layer2)
        self.batchnorm2 = nn.BatchNorm1d(num_features=output_layer2)

        self.linear3 = nn.Linear(output_layer2, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.batchnorm1(x)
        x = nn.Dropout(0.5)(x)

        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = self.batchnorm2(x)
        x = nn.Dropout(0.5)(x)

        x = self.linear3(x)
        x = self.softmax(x)
        return x


class Network_1h(nn.Module):
    def __init__(self, input_size: int, output_layer1: int) -> None:
        super().__init__()

        self.linear1 = nn.Linear(input_size, output_layer1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=output_layer1)

        self.linear2 = nn.Linear(output_layer1, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.batchnorm1(x)
        x = nn.Dropout(0.5)(x)

        x = self.linear2(x)
        x = self.softmax(x)
        return x


class Network_2h(nn.Module):
    def __init__(self, input_size: int, output_layer1: int, output_layer2: int) -> None:
        super().__init__()

        self.linear1 = nn.Linear(input_size, output_layer1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=output_layer1)

        self.linear2 = nn.Linear(output_layer1, output_layer2)
        self.batchnorm2 = nn.BatchNorm1d(num_features=output_layer2)

        self.linear3 = nn.Linear(output_layer2, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.batchnorm1(x)
        x = nn.Dropout(0.5)(x)

        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = self.batchnorm2(x)
        x = nn.Dropout(0.5)(x)

        x = self.linear3(x)
        x = self.softmax(x)
        return x


class Network_3h(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_layer1: int,
        output_layer2: int,
        output_layer3: int,
    ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(input_size, output_layer1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=output_layer1)

        self.linear2 = nn.Linear(output_layer1, output_layer2)
        self.batchnorm2 = nn.BatchNorm1d(num_features=output_layer2)

        self.linear3 = nn.Linear(output_layer2, output_layer3)
        self.batchnorm3 = nn.BatchNorm1d(num_features=output_layer3)

        self.linear4 = nn.Linear(output_layer3, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.batchnorm1(x)
        x = nn.Dropout(0.5)(x)

        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = self.batchnorm2(x)
        x = nn.Dropout(0.5)(x)

        x = self.linear3(x)
        x = nn.ReLU()(x)
        x = self.batchnorm3(x)
        x = nn.Dropout(0.5)(x)

        x = self.linear4(x)
        x = self.softmax(x)
        return x


class Conv(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 4 * 4 + 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Softmax(dim=1),
        )

    def forward(self, data: Tensor) -> Tensor:
        x: Tensor = data[:, :64].view(len(data), 8, 8)
        x = x.unsqueeze(1)
        additional_features = data[:, 64:]

        x = self.conv_layers(x)
        x = flatten(x, 1)
        x = concat((x, additional_features), dim=1)
        x = self.classifier(x)
        return x
