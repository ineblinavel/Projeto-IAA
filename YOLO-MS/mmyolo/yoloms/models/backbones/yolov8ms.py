# Copyright (c) VCIP-NKU. All rights reserved.

import torch
import torch.nn as nn

from mmyolo.models.backbones.base_backbone import BaseBackbone
from mmyolo.models.utils import make_divisible
from mmyolo.registry import MODELS

from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptConfigType

from ..layers.msblock import MSBlock


@MODELS.register_module()
class YOLOv8MS(BaseBackbone):
    """Backbone used in YOLOv8-MS.

    Args:
        arch (str): Architecture of YOLOMS, chosen from predefined configurations 
        (e.g., 'C3-K3579', 'C3-K3579-A234'). Defaults to 'C3-K3579'.
        last_stage_out_channels (int): Number of output channels in the final stage. Defaults to 1024.
        conv_cfg (ConfigType or dict, optional): Configuration for convolution layers. Defaults to None.
        
        msblock_layer_type (str): Type of MSBlock layer. Defaults to "MSBlockBottleNeckLayer".
        msblock_down_ratio (float): Downsampling ratio for MSBlock. Defaults to 1.0.
        msblock_mid_expand_ratio (float): Channel expansion ratio for MSBlock. Defaults to 2.0.
        msblock_layers_num (int): Number of layers in each MSBlock. Defaults to 3.
        msblock_attention_cfg (ConfigType or dict, optional): Configuration for attention mechanisms 
            in MSBlock. Defaults to None.
        msblock_act_cfg (ConfigType): Activation configuration for MSBlock. Defaults to SiLU activation.

        conv_group (str): Grouping strategy for convolutions. Defaults to "auto".
        drop_layer (optional): Dropout layer configuration. Defaults to None.
        spp_config (ConfigType or dict): Configuration for the SPP block. Defaults to 
            dict(type="SPPFBottleneck", kernel_sizes=5).
    
    Returns:
        tuple[Tensor]: Multi-level feature maps from specified output stages.
    """

    arch_settings = {
        'C3-K3579': [
            [MSBlock, 80, 160, [1, (3, 3), (3, 3)], [1, 1, 1], False, False],
            [MSBlock, 160, 320, [1, (5, 5), (5, 5)], [1, 1, 1], False, False],
            [MSBlock, 320, 640, [1, (7, 7), (7, 7)], [1, 1, 1], False, False],
            [MSBlock, 640, 1280, [1, (9, 9), (9, 9)], [1, 1, 1], True, False],
        ],
        'C3-K3579-A234': [
            [MSBlock, 80, 160, [1, (3, 3), (3, 3)], [1, 1, 1], False, False],
            [MSBlock, 160, 320, [1, (5, 5), (5, 5)], [1, 1, 1], False, True],
            [MSBlock, 320, 640, [1, (7, 7), (7, 7)], [1, 1, 1], False, True],
            [MSBlock, 640, 1280, [1, (9, 9), (9, 9)], [1, 1, 1], True, True],
        ],
    }

    def __init__(self,
                 arch: str = 'C3-K3579',
                 last_stage_out_channels: int = 1024,
                 conv_cfg: OptConfigType = None,
                 msblock_layer_type="MSBlockBottleNeckLayer",
                 msblock_down_ratio: float = 1.,
                 msblock_mid_expand_ratio: float = 2.,
                 msblock_layers_num: int = 3,
                 msblock_attention_cfg: OptConfigType = None,
                 msblock_start_branch_id: int = 2,
                 msblock_act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 conv_group="auto",
                 drop_layer=None,
                 spp_config: ConfigType = dict(type="SPPFBottleneck", kernel_sizes=5),
                 **kwargs):
        arch_setting = self.arch_settings[arch]
        self.arch_settings[arch][-1][2] = last_stage_out_channels
        self.conv = ConvModule
        self.conv_group = conv_group
        self.conv_cfg = conv_cfg

        self.msblock_layer_type = msblock_layer_type
        self.msblock_down_ratio = msblock_down_ratio
        self.msblock_mid_expand_ratio = msblock_mid_expand_ratio
        self.msblock_layers_num = msblock_layers_num
        self.msblock_start_branch_id = msblock_start_branch_id
        self.msblock_attention_cfg = msblock_attention_cfg
        self.msblock_act_cfg = msblock_act_cfg

        self.drop_layer = drop_layer
        self.spp_config = spp_config

        super().__init__(self.arch_settings[arch], **kwargs)

        self.msblock_attention_indexs = [
            True if setting[-1] is not None else False for setting in arch_setting
        ]

        self.query = None
        if sum(self.msblock_attention_indexs) != 0:
            assert msblock_attention_cfg is not None, \
                "msblock_attention_cfg should not be None when using branch attention."
            query_length = msblock_attention_cfg.get("length", 3)
            query_size = msblock_attention_cfg.get("size", 4)
            self.query = nn.Parameter(
                torch.randn((query_length, query_size * query_size)),
                requires_grad=True
            ).to(next(self.parameters()).device)

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        return ConvModule(
            self.input_channels,
            make_divisible(self.arch_setting[0][1], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        layer, in_channels, out_channels, kernel_sizes, channel_split_ratios, use_spp, use_attention_cfg = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)

        stage = []
        conv_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        stage.append(conv_layer)

        attention_cfg = self.msblock_attention_cfg if use_attention_cfg else None

        csp_layer = layer(
            out_channels,
            out_channels,
            layer_type=self.msblock_layer_type,
            down_ratio=self.msblock_down_ratio,
            mid_expand_ratio=self.msblock_mid_expand_ratio,
            attention_cfg=attention_cfg,
            kernel_sizes=kernel_sizes,
            channel_split_ratios=channel_split_ratios,
            layers_num=self.msblock_layers_num * self.deepen_factor,
            start_branch_id=self.msblock_start_branch_id,
            conv_group=self.conv_group,
            conv_cfg=self.conv_cfg,
            drop_layer=self.drop_layer,
            act_cfg=self.msblock_act_cfg,
            norm_cfg=self.norm_cfg
        )
        stage.append(csp_layer)

        if use_spp:
            self.spp_config["in_channels"] = out_channels
            self.spp_config["out_channels"] = out_channels
            spp = MODELS.build(self.spp_config)
            stage.append(spp)

        return stage

    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # Reset Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward batch_inputs from the data_preprocessor."""
        outs = []
        for i, layer_name in enumerate(self.layers):
            layers = getattr(self, layer_name)
            if layer_name == 'stem':
                x = layers(x)
            else:
                for layer in layers:
                    if isinstance(layer, MSBlock):
                        x = layer(x, self.query)
                    else:
                        x = layer(x)

            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)