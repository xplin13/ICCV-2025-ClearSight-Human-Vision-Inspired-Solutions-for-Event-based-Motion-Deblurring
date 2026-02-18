import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron
import sys
import os
sys.path.append(os.path.dirname(__file__))
from spikingjelly.activation_based import surrogate, layer
import torch.nn.functional as F
import neuron_gpu


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.GroupNorm(4, out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class NCM(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, lif=True, transpose=False):
        super(NCM, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2

        self.neuron_1 = neuron_gpu.CustomVinitLIFNode(tau=2.,
                                     v_threshold=1.,
                                     v_reset=None,
                                     surrogate_function=surrogate.ATan(),
                                     detach_reset=True,
                                     step_mode = 'm'
                                     )

        self.conv1 = layer.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)

        self.BN1 = layer.BatchNorm2d(out_channel)

    def forward(self, x, v_init_in):

        x = x.permute(1, 0, 2, 3)
        x_conv = self.conv1(x)
        x_bn = self.BN1(x_conv)

        self.neuron_1.v_threshold = 1 - v_init_in.sigmoid()[0]
        lif1, v_out = self.neuron_1(x_bn, v_init=v_init_in)
        out = lif1

        out = out.permute(1, 0, 2, 3)

        return out, v_out


class ResBlock_spike_res_init_cnn(nn.Module):
    def __init__(self, in_channel, out_channel, norm=False):
        super(ResBlock_spike_res_init_cnn, self).__init__()

        self.block1 = NCM(in_channel, out_channel, kernel_size=3, stride=1, lif=True, norm=norm)
        self.block2 = NCM(out_channel, out_channel, kernel_size=3, stride=1, lif=False, norm=norm)

    def forward(self, x, v_init):
        x_ = self.block1(x, v_init)
        x2 = self.block2(x_, v_init)
        return x2 + x
    
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True, norm=norm),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False, norm=norm)
        )

    def forward(self, x):
        return self.main(x) + x