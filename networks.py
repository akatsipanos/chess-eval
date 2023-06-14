import torch.nn as nn

class Network_1h(nn.Module):
    def __init__(self, input_size, output_layer1):
        super(Network_1h, self).__init__()

        self.linear1 = nn.Linear(input_size, output_layer1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=output_layer1)

        self.linear2 = nn.Linear(output_layer1, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.batchnorm1(x)
        x = nn.Dropout(0.5)(x)

        x = self.linear2(x)
        x = self.softmax(x)
        return x

class Network_2h(nn.Module):
    def __init__(self, input_size, output_layer1, output_layer2):
        super(Network_2h, self).__init__()

        self.linear1 = nn.Linear(input_size, output_layer1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=output_layer1)

        self.linear2 = nn.Linear(output_layer1, output_layer2)
        self.batchnorm2 = nn.BatchNorm1d(num_features=output_layer2)

        self.linear3 = nn.Linear(output_layer2, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
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
    def __init__(self, input_size, output_layer1, output_layer2, output_layer3):
        super(Network_3h, self).__init__()

        self.linear1 = nn.Linear(input_size, output_layer1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=output_layer1)

        self.linear2 = nn.Linear(output_layer1, output_layer2)
        self.batchnorm2 = nn.BatchNorm1d(num_features=output_layer2)

        self.linear3 = nn.Linear(output_layer2, output_layer3)
        self.batchnorm3 = nn.BatchNorm1d(num_features=output_layer3)

        self.linear4 = nn.Linear(output_layer3,3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
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