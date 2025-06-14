import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                   padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)

class ShallowPath(nn.Module):
    def __init__(self):
        super(ShallowPath, self).__init__()
        self.conv1 = DepthwiseSeparableConv(1, 16)
        self.conv2 = DepthwiseSeparableConv(16, 32)
        self.conv3 = DepthwiseSeparableConv(32, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class DeepPath(nn.Module):
    def __init__(self):
        super(DeepPath, self).__init__()
        self.conv1 = DepthwiseSeparableConv(1, 16)
        self.conv2 = DepthwiseSeparableConv(16, 32)
        self.conv3 = DepthwiseSeparableConv(32, 64)
        self.conv4 = DepthwiseSeparableConv(64, 96)
        self.conv5 = DepthwiseSeparableConv(96, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class FAModule(nn.Module):
    def __init__(self, in_channels):
        super(FAModule, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        A = self.attn(x)
        return x * A

class LDCE_Net(nn.Module):
    def __init__(self, num_classes):
        super(LDCE_Net, self).__init__()
        self.shallow = ShallowPath()
        self.deep = DeepPath()
        self.fam = FAModule(64 + 128)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64 + 128, num_classes)
        )

    def forward(self, x):
        f1 = self.shallow(x)
        f2 = self.deep(x)
        f_combined = torch.cat((f1, f2), dim=1)
        f_attended = self.fam(f_combined)
        out = self.classifier(f_attended)
        return F.log_softmax(out, dim=1)
