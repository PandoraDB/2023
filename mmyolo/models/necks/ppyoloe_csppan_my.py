# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.models.backbones.csp_resnet import CSPResLayer
from mmyolo.models.necks import BaseYOLONeck
from mmyolo.registry import MODELS
from .transformer_block import *


class ChannelAttention(nn.Module):
    # 传入输入通道数 比例因子ratio(第一次比例因子比较小)
    def __init__(self, in_planes, ratio=8):
        # 初始化
        super(ChannelAttention, self).__init__()

        # 在高宽上进行全局平均池化 把输出高宽设置为1
        # 在高宽上进行全局最大池化 把输出高宽设置为1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 通过两次全连接 第一次输出通道较少 第二层较多 利用1x1卷积代替全连接，类似于SE模块
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, x):
        # 先进行全局平均池化和全局最大池化
        # 再对池化后的结果使用共享的2次全连接层进行处理
        # 再对共享之后的结果进行相加最后再通过sigmoid激活函数
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# 定义CBAM类的空间注意力(CBAM的空间注意力的全局平均池化和全局最大平均池化是对通道进行hxwx1 hxwx1)
class SpatialAttention(nn.Module):
    # 传入卷积核大小
    def __init__(self, kernel_size=7):
        # 初始化
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # 通过7x7卷积
        # 再通过sigmoid激活函数
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, x):
        # 在通道上进行最大池化和平均池化
        # 再进行堆叠(通道上)
        # 堆叠后的结果卷积最后进行sigmoid
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# 定义CBAM类的注意力+空间注意力
class cbam_block(nn.Module):
    # 传入输入通道数 比例因子ratio(第一次比例因子比较小) 传入卷积核大小
    def __init__(self, channel, ratio=8, kernel_size=7):
        # 初始化
        super(cbam_block, self).__init__()
        # 定义通道注意力机制和空间注意力机制
        self.channelattention = ChannelAttention(channel, ratio=ratio)

        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    # 前向传播
    def forward(self, x):
        # x = x*self.channelattention(x)
        # x = x*self.spatialattention(x)

        x_channel = x
        x_spatial = x
        x_channel = x_channel*self.channelattention(x_channel)
        x_spatial = x_spatial*self.spatialattention(x_spatial)
        x = x_spatial + x_channel + x
        return x



@MODELS.register_module()
class PPYOLOECSPPAFPNMY(BaseYOLONeck):
    """CSPPAN in PPYOLOE.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (List[int]): Number of output channels
            (used at each scale).
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        freeze_all(bool): Whether to freeze the model.
        num_csplayer (int): Number of `CSPResLayer` in per layer.
            Defaults to 1.
        num_blocks_per_layer (int): Number of blocks per `CSPResLayer`.
            Defaults to 3.
        block_cfg (dict): Config dict for block. Defaults to
            dict(type='PPYOLOEBasicBlock', shortcut=True, use_alpha=False)
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.1, eps=1e-5).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        drop_block_cfg (dict, optional): Drop block config.
            Defaults to None. If you want to use Drop block after
            `CSPResLayer`, you can set this para as
            dict(type='mmdet.DropBlock', drop_prob=0.1,
            block_size=3, warm_iters=0).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
        use_spp (bool): Whether to use `SPP` in reduce layer.
            Defaults to False.
    """

    def __init__(self,
                 in_channels: List[int] = [256, 512, 1024],
                 out_channels: List[int] = [256, 512, 1024],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 freeze_all: bool = False,
                 num_csplayer: int = 1,
                 num_blocks_per_layer: int = 3,
                 block_cfg: ConfigType = dict(
                     type='PPYOLOEBasicBlock', shortcut=False,
                     use_alpha=False),
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 drop_block_cfg: ConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_spp: bool = False):
        self.block_cfg = block_cfg
        self.num_csplayer = num_csplayer
        self.num_blocks_per_layer = round(num_blocks_per_layer * deepen_factor)
        # Only use spp in last reduce_layer, if use_spp=True.
        self.use_spp = use_spp
        self.drop_block_cfg = drop_block_cfg
        assert drop_block_cfg is None or isinstance(drop_block_cfg, dict)

        super().__init__(
            in_channels=[
                int(channel * widen_factor) for channel in in_channels
            ],
            out_channels=[
                int(channel * widen_factor) for channel in out_channels
            ],
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)


    def build_reduce_layer(self, idx: int):
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        if idx == len(self.in_channels) - 1:
            # fpn_stage
            in_channels = self.in_channels[idx]
            out_channels = self.out_channels[idx]


            layer = [
                CSPResLayer(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    num_block=self.num_blocks_per_layer,
                    block_cfg=self.block_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    attention_cfg=None,
                    use_spp=self.use_spp) for i in range(self.num_csplayer)
            ]

            layer.insert(0, cbam_block(channel=in_channels, kernel_size=3))

            if self.drop_block_cfg:
                layer.append(MODELS.build(self.drop_block_cfg))
            layer = nn.Sequential(*layer)
        else:
            layer = nn.Identity()

        return layer

    def build_upsample_layer(self, idx: int) -> nn.Module:
        """build upsample layer."""
        # fpn_route
        in_channels = self.out_channels[idx]
        return nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            nn.Upsample(scale_factor=2, mode='nearest'))

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        # fpn_stage
        in_channels = self.in_channels[idx - 1] + self.out_channels[idx] // 2
        out_channels = self.out_channels[idx - 1]

        layer = [
            CSPResLayer(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                num_block=self.num_blocks_per_layer,
                block_cfg=self.block_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                attention_cfg=None,
                use_spp=False) for i in range(self.num_csplayer)
        ]

        if self.drop_block_cfg:
            layer.append(MODELS.build(self.drop_block_cfg))

        return nn.Sequential(*layer)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        # pan_route
        return ConvModule(
            in_channels=self.out_channels[idx],
            out_channels=self.out_channels[idx],
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        # pan_stage
        in_channels = self.out_channels[idx + 1] + self.out_channels[idx]
        out_channels = self.out_channels[idx + 1]

        layer = [
            CSPResLayer(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                num_block=self.num_blocks_per_layer,
                block_cfg=self.block_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                attention_cfg=None,
                use_spp=False) for i in range(self.num_csplayer)
        ]
        att = Block(dim=out_channels, num_heads=4)
        layer.append(att)
        if self.drop_block_cfg:
            layer.append(MODELS.build(self.drop_block_cfg))



        return nn.Sequential(*layer)

    def build_out_layer(self, *args, **kwargs) -> nn.Module:
        """build out layer."""
        return nn.Identity()


    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                                                     feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)
        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        # results.insert(0, outs0)
        # results.append(outs_last)

        return tuple(results)