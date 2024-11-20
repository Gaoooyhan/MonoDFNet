# # from __future__ import absolute_import
# # from __future__ import division
# # from __future__ import print_function
#
# import os
# import math
# import logging
# import pdb
# import numpy as np
# from os.path import join
#
# import torch
# from torch import nn
# import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo
# from model.backbone.DCNv2.dcn_v2 import DCN
#
# BN_MOMENTUM = 0.1
#
# def build_backbone(cfg):
#
#     model = DLASeg(base_name=cfg.MODEL.BACKBONE.CONV_BODY,
#                 pretrained=cfg.MODEL.PRETRAIN,
#                 down_ratio=cfg.MODEL.BACKBONE.DOWN_RATIO,
#                 last_level=5,
#             )
#     return model
#
# class DLASeg(nn.Module):
#     def __init__(self, base_name, pretrained, down_ratio, last_level):
#         super(DLASeg, self).__init__()
#         assert down_ratio in [2, 4, 8, 16]
#
#         self.first_level = int(np.log2(down_ratio))
#         self.last_level = last_level
#         self.base = globals()[base_name](pretrained=pretrained)
#
#         channels = self.base.channels
#         scales = [2 ** i for i in range(len(channels[self.first_level:]))]
#         self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)
#
#         self.out_channels = channels[self.first_level]
#
#         self.ida_up = IDAUp(self.out_channels, channels[self.first_level:self.last_level],
#                             [2 ** i for i in range(self.last_level - self.first_level)])
#
#     def forward(self, x):
#         # x: list of features with stride = 1, 2, 4, 8, 16, 32
#         x = self.base(x)
#         x = self.dla_up(x)
#
#         y = []
#         for i in range(self.last_level - self.first_level):
#             y.append(x[i].clone())
#         self.ida_up(y, 0, len(y))
#
#         return y[-1]
#
# def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
#     return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))
#
#
# def conv3x3(in_planes, out_planes, stride=1):
#     "3x3 convolution with padding"
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)
#
#
# class BasicBlock(nn.Module):
#     def __init__(self, inplanes, planes, stride=1, dilation=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
#                                stride=stride, padding=dilation,
#                                bias=False, dilation=dilation)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=dilation,
#                                bias=False, dilation=dilation)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.stride = stride
#
#     def forward(self, x, residual=None):
#         if residual is None:
#             residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
# class Bottleneck(nn.Module):
#     expansion = 2
#
#     def __init__(self, inplanes, planes, stride=1, dilation=1):
#         super(Bottleneck, self).__init__()
#         expansion = Bottleneck.expansion
#         bottle_planes = planes // expansion
#         self.conv1 = nn.Conv2d(inplanes, bottle_planes,
#                                kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
#         self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
#                                stride=stride, padding=dilation,
#                                bias=False, dilation=dilation)
#         self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
#         self.conv3 = nn.Conv2d(bottle_planes, planes,
#                                kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.stride = stride
#
#     def forward(self, x, residual=None):
#         if residual is None:
#             residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class BottleneckX(nn.Module):
#     expansion = 2
#     cardinality = 32
#
#     def __init__(self, inplanes, planes, stride=1, dilation=1):
#         super(BottleneckX, self).__init__()
#         cardinality = BottleneckX.cardinality
#         # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
#         # bottle_planes = dim * cardinality
#         bottle_planes = planes * cardinality // 32
#         self.conv1 = nn.Conv2d(inplanes, bottle_planes,
#                                kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
#         self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
#                                stride=stride, padding=dilation, bias=False,
#                                dilation=dilation, groups=cardinality)
#         self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
#         self.conv3 = nn.Conv2d(bottle_planes, planes,
#                                kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.stride = stride
#
#     def forward(self, x, residual=None):
#         if residual is None:
#             residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class Root(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, residual):
#         super(Root, self).__init__()
#         self.conv = nn.Conv2d(
#             in_channels, out_channels, 1,
#             stride=1, bias=False, padding=(kernel_size - 1) // 2)
#         self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.residual = residual
#
#     def forward(self, *x):
#         children = x
#         x = self.conv(torch.cat(x, 1))
#         x = self.bn(x)
#         if self.residual:
#             x += children[0]
#         x = self.relu(x)
#
#         return x
#
#
# class Tree(nn.Module):
#     def __init__(self, levels, block, in_channels, out_channels, stride=1,
#                  level_root=False, root_dim=0, root_kernel_size=1,
#                  dilation=1, root_residual=False):
#         super(Tree, self).__init__()
#         if root_dim == 0:
#             root_dim = 2 * out_channels
#         if level_root:
#             root_dim += in_channels
#         if levels == 1:
#             self.tree1 = block(in_channels, out_channels, stride,
#                                dilation=dilation)
#             self.tree2 = block(out_channels, out_channels, 1,
#                                dilation=dilation)
#         else:
#             self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
#                               stride, root_dim=0,
#                               root_kernel_size=root_kernel_size,
#                               dilation=dilation, root_residual=root_residual)
#             self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
#                               root_dim=root_dim + out_channels,
#                               root_kernel_size=root_kernel_size,
#                               dilation=dilation, root_residual=root_residual)
#         if levels == 1:
#             self.root = Root(root_dim, out_channels, root_kernel_size,
#                              root_residual)
#         self.level_root = level_root
#         self.root_dim = root_dim
#         self.downsample = None
#         self.project = None
#         self.levels = levels
#         if stride > 1:
#             self.downsample = nn.MaxPool2d(stride, stride=stride)
#         if in_channels != out_channels:
#             self.project = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels,
#                           kernel_size=1, stride=1, bias=False),
#                 nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
#             )
#
#     def forward(self, x, residual=None, children=None):
#         children = [] if children is None else children
#         bottom = self.downsample(x) if self.downsample else x
#         residual = self.project(bottom) if self.project else bottom
#         if self.level_root:
#             children.append(bottom)
#         x1 = self.tree1(x, residual)
#         if self.levels == 1:
#             x2 = self.tree2(x1)
#             x = self.root(x2, x1, *children)
#         else:
#             children.append(x1)
#             x = self.tree2(x1, children=children)
#         return x
#
#
# class DLA(nn.Module):
#     def __init__(self, levels, channels, num_classes=1000,
#                  block=BasicBlock, residual_root=False, linear_root=False):
#         super(DLA, self).__init__()
#         self.channels = channels
#         self.num_classes = num_classes
#         self.base_layer = nn.Sequential(
#             nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
#                       padding=3, bias=False),
#             nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
#             nn.ReLU(inplace=True))
#         self.level0 = self._make_conv_level(
#             channels[0], channels[0], levels[0])
#         self.level1 = self._make_conv_level(
#             channels[0], channels[1], levels[1], stride=2)
#         self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
#                            level_root=False,
#                            root_residual=residual_root)
#         self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
#                            level_root=True, root_residual=residual_root)
#         self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
#                            level_root=True, root_residual=residual_root)
#         self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
#                            level_root=True, root_residual=residual_root)
#
#         # for m in self.modules():
#         #     if isinstance(m, nn.Conv2d):
#         #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         #         m.weight.data.normal_(0, math.sqrt(2. / n))
#         #     elif isinstance(m, nn.BatchNorm2d):
#         #         m.weight.data.fill_(1)
#         #         m.bias.data.zero_()
#
#     def _make_level(self, block, inplanes, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or inplanes != planes:
#             downsample = nn.Sequential(
#                 nn.MaxPool2d(stride, stride=stride),
#                 nn.Conv2d(inplanes, planes,
#                           kernel_size=1, stride=1, bias=False),
#                 nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
#             )
#
#         layers = []
#         layers.append(block(inplanes, planes, stride, downsample=downsample))
#         for i in range(1, blocks):
#             layers.append(block(inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
#         modules = []
#         for i in range(convs):
#             modules.extend([
#                 nn.Conv2d(inplanes, planes, kernel_size=3,
#                           stride=stride if i == 0 else 1,
#                           padding=dilation, bias=False, dilation=dilation),
#                 nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
#                 nn.ReLU(inplace=True)])
#             inplanes = planes
#         return nn.Sequential(*modules)
#
#     def forward(self, x):
#         y = []
#         x = self.base_layer(x)
#         for i in range(6):
#             x = getattr(self, 'level{}'.format(i))(x)
#             y.append(x)
#
#         return y
#
#     def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
#         # fc = self.fc
#         if name.endswith('.pth'):
#             model_weights = torch.load(data + name)
#         else:
#             model_url = get_model_url(data, name, hash)
#             model_weights = model_zoo.load_url(model_url)
#         num_classes = len(model_weights[list(model_weights.keys())[-1]])
#         self.fc = nn.Conv2d(
#             self.channels[-1], num_classes,
#             kernel_size=1, stride=1, padding=0, bias=True)
#         self.load_state_dict(model_weights)
#
#
# def dla34(pretrained=True, **kwargs):  # DLA-34
#     model = DLA([1, 1, 1, 2, 2, 1],
#                 [16, 32, 64, 128, 256, 512],
#                 block=BasicBlock, **kwargs)
#     if pretrained:
#         model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
#
#     return model
#
# class Identity(nn.Module):
#
#     def __init__(self):
#         super(Identity, self).__init__()
#
#     def forward(self, x):
#         return x
#
#
# def fill_fc_weights(layers):
#     for m in layers.modules():
#         if isinstance(m, nn.Conv2d):
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#
#
# def fill_up_weights(up):
#     w = up.weight.data
#     f = math.ceil(w.size(2) / 2)
#     c = (2 * f - 1 - f % 2) / (2. * f)
#     for i in range(w.size(2)):
#         for j in range(w.size(3)):
#             w[0, 0, i, j] = \
#                 (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
#     for c in range(1, w.size(0)):
#         w[c, 0, :, :] = w[0, 0, :, :]
#
#
# class DeformConv(nn.Module):
#     def __init__(self, chi, cho):
#         super(DeformConv, self).__init__()
#         self.actf = nn.Sequential(
#             nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
#             nn.ReLU(inplace=True)
#         )
#         self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.actf(x)
#         return x
#
#
# class IDAUp(nn.Module):
#
#     def __init__(self, o, channels, up_f):
#         super(IDAUp, self).__init__()
#         for i in range(1, len(channels)):
#             c = channels[i]
#             f = int(up_f[i])
#             proj = DeformConv(c, o)
#             node = DeformConv(o, o)
#
#             up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
#                                     padding=f // 2, output_padding=0,
#                                     groups=o, bias=False)
#             fill_up_weights(up)
#
#             setattr(self, 'proj_' + str(i), proj)
#             setattr(self, 'up_' + str(i), up)
#             setattr(self, 'node_' + str(i), node)
#
#
#     def forward(self, layers, startp, endp):
#         for i in range(startp + 1, endp):
#             upsample = getattr(self, 'up_' + str(i - startp))
#             project = getattr(self, 'proj_' + str(i - startp))
#             layers[i] = upsample(project(layers[i]))
#             node = getattr(self, 'node_' + str(i - startp))
#             layers[i] = node(layers[i] + layers[i - 1])
#
#
#
# class DLAUp(nn.Module):
#     def __init__(self, startp, channels, scales, in_channels=None):
#         super(DLAUp, self).__init__()
#         self.startp = startp
#         if in_channels is None:
#             in_channels = channels
#         self.channels = channels
#         channels = list(channels)
#         scales = np.array(scales, dtype=int)
#         for i in range(len(channels) - 1):
#             j = -i - 2
#             setattr(self, 'ida_{}'.format(i),
#                     IDAUp(channels[j], in_channels[j:],
#                           scales[j:] // scales[j]))
#             scales[j + 1:] = scales[j]
#             in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]
#
#     def forward(self, layers):
#         out = [layers[-1]] # start with 32
#         for i in range(len(layers) - self.startp - 1):
#             ida = getattr(self, 'ida_{}'.format(i))
#             ida(layers, len(layers) -i - 2, len(layers))
#             out.insert(0, layers[-1])
#         return out
#
#
# class Interpolate(nn.Module):
#     def __init__(self, scale, mode):
#         super(Interpolate, self).__init__()
#         self.scale = scale
#         self.mode = mode
#
#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
#         return x
#
# if __name__ == '__main__':
#     model = build_backbone(num_layers=34).cuda()
#     x = torch.rand(2, 3, 384, 1280).cuda()
#     y = model(x)
#
#     print(y.shape)
#
#
#
#



# --------------------------增加空间注意力机制
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import math
import logging
import pdb
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from model.backbone.DCNv2.dcn_v2 import DCN
import torchvision.transforms.functional as TF
import cv2

BN_MOMENTUM = 0.1

import torch
from torch import nn
import torch.nn.functional as F
import torch
from torch import nn

# class SpatialAttentionModule(nn.Module):
#     def __init__(self, kernel_size=9):
#         super(SpatialAttentionModule, self).__init__()
#         self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.alpha = nn.Parameter(torch.ones(1))  # 可学习的权重参数
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         attention = self.sigmoid(x)
#         return x * self.alpha + attention * (1 - self.alpha)  # 原始特征与注意力特征的加权和
# class MultiBranchDepthPredictor(nn.Module):
#     def __init__(self, in_channels, out_channels=1):
#         super(MultiBranchDepthPredictor, self).__init__()
#         # 定义共享卷积层
#         self.shared_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
#         self.shared_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
#
#         # 定义每个分支的卷积
#         self.branch1_conv = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # 分支 1
#         self.branch2_conv = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # 分支 2
#         self.branch3_conv = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # 分支 3
#
#         # 定义每个分支的深度预测头
#         self.branch1_depth = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)  # 分支 1 深度图
#         self.branch2_depth = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)  # 分支 2 深度图
#         self.branch3_depth = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)  # 分支 3 深度图
#
#         # 可学习的权重参数
#         self.branch1_weight = nn.Parameter(torch.tensor(0.33))  # 初始化权重
#         self.branch2_weight = nn.Parameter(torch.tensor(0.33))
#         self.branch3_weight = nn.Parameter(torch.tensor(0.34))
#
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         # 通过共享卷积层提取特征
#         x = self.relu(self.shared_conv1(x))
#         x = self.relu(self.shared_conv2(x))
#
#         # 每个分支的特征处理
#         branch1_feature = self.relu(self.branch1_conv(x))
#         branch2_feature = self.relu(self.branch2_conv(x))
#         branch3_feature = self.relu(self.branch3_conv(x))
#
#         # 每个分支的深度图预测
#         branch1_depth = self.branch1_depth(branch1_feature)
#         branch2_depth = self.branch2_depth(branch2_feature)
#         branch3_depth = self.branch3_depth(branch3_feature)
#
#         # 加权融合
#         fused_depth = (self.branch1_weight * branch1_depth +
#                        self.branch2_weight * branch2_depth +
#                        self.branch3_weight * branch3_depth) / (self.branch1_weight + self.branch2_weight + self.branch3_weight)
#
#         return fused_depth
class MultiBranchDepthPredictor(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(MultiBranchDepthPredictor, self).__init__()
        # 定义共享卷积层
        self.shared_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.shared_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # 定义每个分支的卷积
        self.branch1_conv = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # 分支 1
        self.branch2_conv = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # 分支 2
        self.branch3_conv = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # 分支 3

        # 定义每个分支的深度预测头
        self.branch1_depth = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)  # 分支 1 深度图
        self.branch2_depth = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)  # 分支 2 深度图
        self.branch3_depth = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)  # 分支 3 深度图

        # 可学习的加权参数
        self.alpha1 = nn.Parameter(torch.tensor(0.9))  # 分支 1 权重
        self.alpha2 = nn.Parameter(torch.tensor(1.0))  # 分支 2 权重
        self.alpha3 = nn.Parameter(torch.tensor(1.1))  # 分支 3 权重

        self.relu = nn.ReLU()

    def forward(self, x):
        # 通过共享卷积层提取特征
        x = self.relu(self.shared_conv1(x))
        x = self.relu(self.shared_conv2(x))

        # 每个分支的特征处理
        branch1_feature = self.relu(self.branch1_conv(x))
        branch2_feature = self.relu(self.branch2_conv(x))
        branch3_feature = self.relu(self.branch3_conv(x))

        # 每个分支的深度图预测
        branch1_depth = self.branch1_depth(branch1_feature)
        branch2_depth = self.branch2_depth(branch2_feature)
        branch3_depth = self.branch3_depth(branch3_feature)

        # 使用 Softmax 正则化 alpha 权重
        weights = torch.softmax(torch.stack([self.alpha1, self.alpha2, self.alpha3]), dim=0)

        # 使用正则化后的权重进行分支深度融合
        fused_depth = weights[0] * branch1_depth + weights[1] * branch2_depth + weights[2] * branch3_depth
        return fused_depth


class DepthPredictor(nn.Module):
    def __init__(self, in_channels=256, out_channels=1):  # 修改 in_channels 为 256
        super(DepthPredictor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # 残差连接
        self.residual1 = nn.Conv2d(in_channels, 64, kernel_size=1)  # 修改 in_channels 为 256
        self.residual2 = nn.Conv2d(64, 32, kernel_size=1)

    def forward(self, x):
        # 第一个卷积块的残差
        residual1 = self.residual1(x)
        x = self.relu(self.conv1(x)) + residual1

        # 第二个卷积块的残差
        residual2 = self.residual2(x)
        x = self.relu(self.conv2(x)) + residual2

        # 输出深度图
        depth_map = self.conv3(x)
        return depth_map


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=9):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.ones(1))  # 可学习的权重参数

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_map = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.conv1(attention_map)
        attention_map = self.sigmoid(attention_map)
        return x * self.alpha + attention_map * (1 - self.alpha)  # 保持输入的通道数不变


def build_backbone(cfg):

    model = DLASeg(base_name=cfg.MODEL.BACKBONE.CONV_BODY,
                pretrained=cfg.MODEL.PRETRAIN,
                down_ratio=cfg.MODEL.BACKBONE.DOWN_RATIO,
                last_level=5,
            )
    return model

class DLASeg(nn.Module):
    def __init__(self, base_name, pretrained, down_ratio, last_level):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]

        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)

        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        self.out_channels = channels[self.first_level]

        self.ida_up = IDAUp(self.out_channels, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        # 添加空间注意力模块
        self.spatial_attention = SpatialAttentionModule()

        # 使用多尺度深度预测模块
        self.depth_predictor = MultiBranchDepthPredictor(in_channels=256, out_channels=1)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())

        # 应用空间注意力机制到特征图
        attention_feature = self.spatial_attention(y[-1])

        # 深度预测支路（不影响原有特征图的流动）
        depth_map = self.depth_predictor(attention_feature.clone())

        # IDA Up 操作（用于 2D 和 3D 目标检测）
        self.ida_up(y, 0, len(y))

        # 返回原有的特征图和深度图
        return y[-1], depth_map



    # def forward(self, x):
    #     x = self.base(x)
    #     x = self.dla_up(x)
    #
    #     y = []
    #     for i in range(self.last_level - self.first_level):
    #         y.append(x[i].clone())
    #
    #     y[-1] = y[-1] * self.spatial_attention(y[-1])
    #     self.ida_up(y, 0, len(y))
    #
    #     return y[-1]

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)

        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')

    return model

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)


    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])



class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x

if __name__ == '__main__':
    model = build_backbone(num_layers=34).cuda()
    x = torch.rand(2, 3, 384, 1280).cuda()
    y = model(x)

    print(y.shape)
