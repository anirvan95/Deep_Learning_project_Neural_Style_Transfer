import torch
from transformer_net import ConvLayer, UpsampleConvLayer, ResidualBlock
from collections import namedtuple


class SmallTransformerNet(torch.nn.Module):
    def __init__(self):
        super(SmallTransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 64, kernel_size=9, stride=2)
        self.in1 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv2 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=4)
        self.in3 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = ConvLayer(64, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y_relu_1 = y;
        y = self.res1(y)
        y = self.res2(y)
        y_relu_2 = y;
        y = self.relu(self.in3(self.deconv1(y)))
        y = self.deconv2(y)
        y_relu_3 = y;
        SmallTransformerNet_outputs = namedtuple("SmallTransformerNetOutputs", ['relu_1', 'relu_2', 'relu_3'])
        out = SmallTransformerNet_outputs(y_relu_1, y_relu_2, y_relu_3)
        return out


