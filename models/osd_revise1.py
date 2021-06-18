# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from models.ghost_net import GhostModule



def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False, ):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        # self.add_module('conv1',GhostModule(num_input_features, bn_size * growth_rate,
        #                 kernel_size=1, stride=1)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                       kernel_size=1, stride=1, bias=False)),


        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', HardSwish(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.add_module('conv22', GhostModule(bn_size * growth_rate, growth_rate,
                                             kernel_size=3, stride=1)),
        self.add_module('conv222', DWConv(bn_size * growth_rate, growth_rate, stride=1))
        self.add_module('shuffle', channel_shuffle())
        self.drop_rate = drop_rate
        self.efficient = efficient
    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        #new_features= self.conv2(self.relu2(self.norm2(bottleneck_output)))+self.conv22(self.relu2(self.norm2(bottleneck_output)))
        new_features_t = torch.cat([self.conv2(self.relu2(self.norm2(bottleneck_output))) , self.conv22(self.relu2(self.norm2(bottleneck_output)))],1)
        new_features = torch.cat([new_features_t, self.conv222(self.relu2(self.norm2(bottleneck_output)))], 1)
        new_features = self.shuffle(new_features)
        #new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        #new_features=bottleneck_output
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseLayer1(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False, ):
        super(_DenseLayer1, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        # self.add_module('conv1',GhostModule(num_input_features, bn_size * growth_rate,
        #                 kernel_size=1, stride=1)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                       kernel_size=1, stride=1, bias=False)),

        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', HardSwish(inplace=True)),
        # self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
        #                                    kernel_size=3, stride=1, padding=1, bias=False)),
        # self.add_module('conv22', GhostModule(bn_size * growth_rate, growth_rate,
        #                                      kernel_size=3, stride=1)),
        self.add_module('conv222', DWConv(bn_size * growth_rate, growth_rate, stride=1))
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        #new_features= self.conv2(self.relu2(self.norm2(bottleneck_output)))+self.conv22(self.relu2(self.norm2(bottleneck_output)))
        new_features = self.conv222(self.relu2(self.norm2(bottleneck_output)))
        # new_features_t = torch.cat([self.conv2(self.relu2(self.norm2(bottleneck_output))) , self.conv22(self.relu2(self.norm2(bottleneck_output)))],1)
        # new_features = torch.cat([new_features_t, self.conv222(self.relu2(self.norm2(bottleneck_output)))], 1)
        #new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        #new_features=bottleneck_output
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))

        self.add_module('conv', DWConv(num_input_features, num_output_features, stride=1)),
        # self.add_module('conv', GhostModule(num_input_features, num_output_features,
        #                                      kernel_size=1, stride=1)),
        # self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
        #                                  kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
# class _Transition1(nn.Sequential):
#     def __init__(self, num_input_features):
#         super(_Transition1, self).__init__()
#         # self.add_module('norm', nn.BatchNorm2d(num_input_features))
#         # self.add_module('relu', nn.ReLU(inplace=True))
#         #
#         # self.add_module('conv', DWConv(num_input_features, num_output_features, stride=1)),
#         # #self.add_module('conv', GhostModule(num_input_features, num_output_features,
#         #                                      kernel_size=1, stride=1)),
#         #self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
#         #                                  kernel_size=1, stride=1, bias=False))
#         self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + 3 * i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            #print(name)
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _DenseBlock1(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock1, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer1(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            #print(name)
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DWConv, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


def hard_sigmoid(x: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    return nn.functional.relu6(x + 3, inplace=inplace) / 6
def hard_swish(x: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    return hard_sigmoid(x, inplace=inplace) * x
class HardSwish(nn.Module):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self._inplace = True
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hard_swish(x, inplace=self._inplace)
def Channel_shuffle(inp, n_groups=3):
    batch_size, chans, height, width = inp.data.size()
    chans_group = chans // n_groups
    # reshape
    inp = inp.view(batch_size, n_groups, chans_group, height, width)
    inp = torch.transpose(inp, 1, 2).contiguous()
    inp = inp.view(batch_size, -1, height, width)
    return inp
class channel_shuffle(nn.Module):
    def __init__(self):
        super(channel_shuffle, self).__init__()
    def forward(self, x):
        return Channel_shuffle(x)


class First_layer(nn.Module):
    def __init__(self, num_init_features, drop_rate, efficient=False ):
        super(First_layer, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=2, padding=1, bias=False)),
            ('conv00', GhostModule(3, num_init_features, kernel_size=3, stride=2)),
            ('conv000', DWConv(3, num_init_features, stride=2)),
        ]))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features * 3))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        # self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
        #                                                ceil_mode=False))
        self.features.add_module('shuffle0', channel_shuffle())

        self.drop_rate = drop_rate
        self.efficient = efficient
    def forward(self, x):
        x1 = self.features.conv0(x)
        x2 = self.features.conv00(x)
        x3 = self.features.conv000(x)
        x4 = torch.cat([x1, x2], 1)
        x5 = torch.cat([x4, x3], 1)
        x6 = self.features.norm0(x5)
        x7 = self.features.relu0(x6)
        # x8 = self.features.pool0(x7)
        x9 = self.features.shuffle0(x7)
        return x9

class OsdNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=12, block_config=(2, 4, 6), compression=0.8,
                 num_init_features=24, bn_size=2, drop_rate=0,
                 num_classes=10, efficient=True):
        super(OsdNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        # First convolution
        # self.conv0 = nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
        # self.conv00 = GhostModule(3, num_init_features, kernel_size=7, stride=2),
        # self.conv000 = DWConv(3, num_init_features, stride=2),
        # #连接为54，
        # self.norm0 = nn.BatchNorm2d(num_init_features*3)
        # self.relu0 = HardSwish(inplace=True)
        # self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        # #通道打乱
        # self.shuffle0 = channel_shuffle
        self.features = nn.Sequential(OrderedDict([
                    ('first_layer', First_layer(num_init_features, drop_rate)),
                ]))

        # if small_inputs:
        #     self.features = nn.Sequential(OrderedDict([
        #         ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
        #     ]))
        # else:
        #     self.features = nn.Sequential(OrderedDict([
        #         ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=2, padding=1, bias=False)),
        #         ('conv00', GhostModule(3, num_init_features, kernel_size=3, stride=2)),
        #         ('conv000', DWConv(3, num_init_features, stride=2)),
        #     ]))
        #     self.features.add_module('norm0', nn.BatchNorm2d(num_init_features*3))
        #     self.features.add_module('relu0', HardSwish(inplace=True))
        #     self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
        #                                                    ceil_mode=False))
        #     self.features.add_module('shuffle0', channel_shuffle())

        # Each denseblock
        num_features = num_init_features*3
        for i, num_layers in enumerate(block_config):
            if i != 5:
                block = _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    efficient=efficient,
                )
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + 3 * num_layers * growth_rate
                if i == 0:
                    self.features.add_module('shuffle%d' % (i+1) , channel_shuffle())
                if i != len(block_config) - 1:
                    trans = _Transition(num_input_features=num_features,
                                        num_output_features=int(num_features * compression))
                    self.features.add_module('transition%d' % (i + 1), trans)
                    num_features = int(num_features * compression)
            else:
                block = _DenseBlock1(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    efficient=efficient,
                )
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate   #108+12*4=156
                self.features.add_module('shuffle%d' % (i+1) , channel_shuffle())
                if i != len(block_config) - 1:
                    trans = _Transition(num_input_features=num_features,
                                        num_output_features=int(num_features * compression))
                    self.features.add_module('transition%d' % (i + 1), trans)
                    num_features = int(num_features * compression)

        # Final batch norm
        self.conv5 = nn.Sequential(
            nn.Conv2d(num_features, num_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
        )

        #self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        for name, param in self.named_parameters():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            if 'conv' in name and 'weight' in name:
                #n = param.size(0) * param.size(2) * param.size(3)
                n=2
                #param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        # x1 = self.conv0(x)
        # x2 = self.conv00(x)
        # x3 = self.conv000(x)
        # x4 = torch.cat([x1, x2], 1)
        # x5 = torch.cat([x4, x3], 1)
        # x6 = self.norm0(x5)
        # x7 = self.relu0(x6)
        # x8 = self.pool0(x7)
        # x = self.shuffle0(x8)
        features = self.features(x)
        inp = self.conv5(features)
        inp = inp.mean([2, 3])  # globalpool
        inp = inp.view(inp.size(0), -1)
        out = self.classifier(inp)
        return out

if __name__ == "__main__":
    net = OsdNet()
    print(net)
    a = torch.randn(1, 3, 672, 512)
    b = net(a)
    print(b.shape)
