import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """LeNet-5 without BatchNorm, adapted for 64x64 RGB inputs."""

    def __init__(self, num_classes: int = 10):
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(1, -1),
            
            nn.Linear(16 * 13 * 13, 120),
            nn.Tanh(),
            
            nn.Linear(120, 84),
            nn.Tanh(),
            
            nn.Linear(84, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

    def feature_layer(self):
        return self.model[1]


class LeNetBN(nn.Module):
    """LeNet-5 with BatchNorm2d after each convolution."""

    def __init__(self, num_classes: int = 10):
        super(LeNetBN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(1, -1),

            nn.Linear(16 * 13 * 13, 120),
            nn.Tanh(),

            nn.Linear(120, 84),
            nn.Tanh(),

            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def feature_layer(self):
        return self.model[2]


class ScaledTanh(nn.Module):
    def __init__(self, amplitude: float = 1.7159, slope: float = 2.0 / 3.0):
        super().__init__()
        self.amplitude = amplitude
        self.slope = slope

    def forward(self, x):
        return self.amplitude * torch.tanh(self.slope * x)


class TrainableSubsampling2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.full((1, channels, 1, 1), 0.25))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.activation = ScaledTanh()

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=2, stride=2) * 4.0
        x = x * self.weight + self.bias
        return self.activation(x)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, connection_table):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size)
        mask = torch.zeros(out_channels, in_channels, 1, 1)
        for out_idx, input_maps in enumerate(connection_table):
            mask[out_idx, input_maps, 0, 0] = 1.0
        self.register_buffer("connection_mask", mask)

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight * self.connection_mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class RBFOutput(nn.Module):
    def __init__(self, in_features: int, num_classes: int, seed: int = 42):
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(seed)
        centers = torch.randint(0, 2, (num_classes, in_features), generator=generator).float()
        centers = centers.mul_(2.0).sub_(1.0)
        self.register_buffer("centers", centers)

    def forward(self, x):
        return (x.unsqueeze(1) - self.centers.unsqueeze(0)).pow(2).sum(dim=2)


class LeNet5Paper(nn.Module):
    """LeNet-5 close to LeCun et al. (1998), adapted to num_classes."""

    c3_connections = [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [0, 4, 5],
        [0, 1, 5],
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [0, 3, 4, 5],
        [0, 1, 4, 5],
        [0, 1, 2, 5],
        [0, 1, 3, 4],
        [1, 2, 4, 5],
        [0, 2, 3, 5],
        [0, 1, 2, 3, 4, 5],
    ]

    def __init__(self, num_classes: int = 10, output: str = "rbf"):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        self.s2 = TrainableSubsampling2d(6)
        self.c3 = MaskedConv2d(6, 16, kernel_size=5, connection_table=self.c3_connections)
        self.s4 = TrainableSubsampling2d(16)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.f6 = nn.Linear(120, 84)
        self.activation = ScaledTanh()
        if output == "rbf":
            self.output = RBFOutput(84, num_classes)
        elif output == "linear":
            self.output = nn.Linear(84, num_classes)
        else:
            raise ValueError("output must be 'rbf' or 'linear'")
        self.output_kind = output
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, MaskedConv2d, nn.Linear)):
                fan_in = nn.init._calculate_correct_fan(module.weight, "fan_in")
                nn.init.uniform_(module.weight, -2.4 / fan_in, 2.4 / fan_in)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.activation(self.c1(x))
        x = self.s2(x)
        x = self.activation(self.c3(x))
        x = self.s4(x)
        x = self.activation(self.c5(x))
        x = torch.flatten(x, 1)
        x = self.activation(self.f6(x))
        return self.output(x)


class LeNet5MAPLoss(nn.Module):
    def __init__(self, rubbish_penalty: float = 0.1):
        super().__init__()
        self.rubbish_penalty = rubbish_penalty

    def forward(self, penalties, target):
        true_penalty = penalties.gather(1, target.view(-1, 1)).squeeze(1)
        competition = torch.logsumexp(
            torch.cat(
                [
                    penalties.new_full((penalties.size(0), 1), -self.rubbish_penalty),
                    -penalties,
                ],
                dim=1,
            ),
            dim=1,
        )
        return (true_penalty + competition).mean()
