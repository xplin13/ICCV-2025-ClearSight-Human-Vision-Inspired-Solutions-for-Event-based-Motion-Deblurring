import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from einops import rearrange
import numbers
from torchvision.ops import DeformConv2d


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mutual_AttentionwithFC(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_AttentionwithFC, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.scale = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.shift = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        b, c, h, w = x.shape

        scale = F.leaky_relu(self.scale(x))
        shift = F.leaky_relu(self.shift(x))
        y_ = y * (scale + 1) + shift

        q = self.q(x)  # image
        k = self.k(y_)  # event
        v = self.v(y_)  # event

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out


class RBAM(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(RBAM, self).__init__()

        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_AttentionwithFC(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim * 2)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim * 2, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.conv = nn.Conv2d(in_channels=2 * dim, out_channels=dim, kernel_size=1, stride=1)

        self.threshold_mlp = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1),
        )

        # 可变形卷积层
        self.deform_conv = DeformConv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # 初始化权重为1且不训练
        self.deform_conv.weight.data.fill_(1.0)
        self.deform_conv.weight.requires_grad = False

        # 生成偏移量的卷积层
        self.offset_conv = nn.Conv2d(dim, 18, kernel_size=3, padding=1, stride=1)

    def forward(self, image, event, spike_map):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, 2c, h, w
        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c, h, w = image.shape

        threshold = torch.abs(self.threshold_mlp(image))  # 输出尺寸: (b, 1, h, w)
        threshold_max = torch.max(threshold)
        threshold_min = torch.min(threshold)
        threshold = (threshold - threshold_min) / (threshold_max - threshold_min + 1e-10)

        spike_sum_0 = spike_map.sum(dim=1, keepdim=True)
        spike_sum_1 = spike_sum_0.sum(dim=0, keepdim=True)

        # 从图像特征生成偏移量
        offset = self.offset_conv(image)  # 使用图像数据而非脉冲数据生成偏移量

        # 使用可变形卷积和偏移量
        local_sum = self.deform_conv(spike_sum_1, offset)

        spike_max = torch.max(local_sum)
        spike_min = torch.min(local_sum)
        spike_sum = (local_sum - spike_min) / (spike_max - spike_min + 1e-10)

        basic_mask = (spike_sum > threshold).float()

        spike_zero_mask = (spike_sum_0 == 0).any(dim=0).float()

        mask = basic_mask * (1 - spike_zero_mask)
        mask = mask.permute(1, 0, 2, 3)

        fused_image = image + mask * self.attn(self.norm1_image(image), self.norm1_event(event))
        fused_event = event + (1-mask) * self.attn(self.norm1_image(event), self.norm1_event(image))
        # mlp
        fused = torch.cat([fused_image, fused_event], dim=1)
        fused = to_3d(fused)  # b, h*w, 2c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)  # b,2c,h,w
        fused = self.conv(fused)

        return fused, mask