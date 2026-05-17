import torch.nn as nn


VGG11_HALF_CONFIG = [32, "M", 64, "M", 128, 128, "M", 256, 256, "M", 256, 256, "M"]


def make_vgg_layers(config, batch_norm: bool = False):
    layers = []
    in_channels = 3
    for layer in config:
        if layer == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue

        conv2d = nn.Conv2d(in_channels, layer, kernel_size=3, padding=1)
        if batch_norm:
            layers.extend([conv2d, nn.BatchNorm2d(layer), nn.ReLU(inplace=True)])
        else:
            layers.extend([conv2d, nn.ReLU(inplace=True)])
        in_channels = layer
    return nn.Sequential(*layers)


class VGG11Half(nn.Module):
    def __init__(self, num_classes: int = 10, batch_norm: bool = False):
        super(VGG11Half, self).__init__()
        self._bn = batch_norm
        self.features = make_vgg_layers(VGG11_HALF_CONFIG, batch_norm=batch_norm)
        self.classifier = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def feature_layer(self):
        return self.features[2] if self._bn else self.features[1]


class VGG11HalfBN(VGG11Half):
    def __init__(self, num_classes: int = 10):
        super(VGG11HalfBN, self).__init__(num_classes=num_classes, batch_norm=True)
