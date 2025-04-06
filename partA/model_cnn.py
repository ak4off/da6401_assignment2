# model_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, 
                 num_classes=10, 
                 num_filters=[32, 64, 128, 128, 256], 
                 filter_sizes=[3, 3, 3, 3, 3],
                 activation='relu',
                 dense_neurons=512,
                 dropout_rate=0.5,
                 batch_norm=True,
                 input_size=128):
        super(CNN, self).__init__()
        self.activation_fn = self._get_activation(activation)
        self.batch_norm = batch_norm

        layers = []
        in_channels = 3  # for RGB images
        current_size = input_size

        for i in range(5):
            layers.append(nn.Conv2d(in_channels, num_filters[i], kernel_size=filter_sizes[i], padding=1))
            if self.batch_norm:
                layers.append(nn.BatchNorm2d(num_filters[i]))
            layers.append(self.activation_fn)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_channels = num_filters[i]
            current_size = current_size // 2  # halved each time due to maxpool

        self.conv = nn.Sequential(*layers)

        self.flattened_size = num_filters[-1] * current_size * current_size

        self.fc1 = nn.Linear(self.flattened_size, dense_neurons)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(dense_neurons, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

    def _get_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError("Unsupported activation function")

    def compute_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_flops(self, input_size=(3, 128, 128)):
        flops = 0
        in_channels = input_size[0]
        h, w = input_size[1], input_size[2]

        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                out_channels = layer.out_channels
                kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
                out_h = h // 2
                out_w = w // 2
                flops += 2 * in_channels * out_channels * kernel_size * out_h * out_w
                h, w = out_h, out_w
                in_channels = out_channels

        # Dense layer FLOPs
        flops += 2 * self.flattened_size * self.fc1.out_features
        flops += 2 * self.fc1.out_features * self.fc_out.out_features
        return flops
