# Refer to MMYOLO
# Copyright (c) VCIP-NKU. All rights reserved.

from typing import Sequence, Union

import torch.nn as nn

from mmyolo.registry import MODELS
from mmyolo.models.necks.yolov8_pafpn import YOLOv8PAFPN
from mmyolo.models.utils import make_divisible
from mmdet.utils import ConfigType, OptConfigType

from ..layers import MSBlock


@MODELS.register_module()
class YOLOv8MSPAFPN(YOLOv8PAFPN):
    """Path Aggregation Network in YOLOv8 with MS-Blocks.

    Args:
        deepen_factor (float): Depth multiplier, scales the number of blocks in the network. Defaults to 1.0.
        widen_factor (float): Width multiplier, scales the number of channels in each layer. Defaults to 1.0.
        kernel_sizes (Sequence[Union[int, Sequence[int]]]): Kernel sizes for the MS-Block layers. Defaults to [1, 3, 3].

        msblock_layer_type (str): The type of layer used in MS-Block. Defaults to "MSBlockBottleNeckLayer".
        msblock_down_ratio (float): Channel down ratio for the downsample convolution layer in MS-Block. Defaults to 1.
        msblock_mid_expand_ratio (float): Channel expansion ratio for each branch in MS-Block. Defaults to 2.
        msblock_layers_num (int): Number of layers in the MS-Block. Defaults to 3.
        msblock_channel_split_ratios (list[float]): Ratios for splitting channels among branches in MS-Block. Defaults to [1, 1, 1].
        msblock_attention_cfg (:obj:`ConfigDict` or dict, optional): Configuration dictionary for attention in MS-Block. Defaults to None.
        msblock_act_cfg (:obj:`ConfigDict` or dict): Configuration dictionary for activation layers in MS-Block. Defaults to `dict(type="SiLU", inplace=True)`.

        conv_group (str): Grouping method for convolutions. Defaults to "auto".
        conv_cfg (:obj:`ConfigDict` or dict, optional): Configuration dictionary for convolution layers. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Configuration dictionary for normalization layers. Defaults to `dict(type="BN")`.
        drop_layer (:obj:`ConfigDict` or dict, optional): Configuration dictionary for dropout layers. Defaults to None.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        kernel_sizes: Sequence[Union[int, Sequence[int]]] = [1, 3, 3],

        msblock_layer_type="MSBlockBottleNeckLayer",
        msblock_down_ratio: float = 1,
        msblock_mid_expand_ratio: float = 2,
        msblock_layers_num: int = 3,
        msblock_start_branch_id: int = 2,
        msblock_channel_split_ratios=[1, 1, 1],
        msblock_attention_cfg: OptConfigType = None,
        msblock_act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        
        conv_group="auto",
        conv_cfg: bool = None,
        norm_cfg: ConfigType = dict(type="BN"),
        drop_layer=None,
        **kwargs,
    ):
        if isinstance(kernel_sizes, list):
            kernel_sizes = {
                "bottom_up": [kernel_sizes for _ in range(2)],
                "top_down": [kernel_sizes for _ in range(2)],
            }
        self.kernel_sizes = kernel_sizes
        self.layer_config = dict(
            layer_type=msblock_layer_type,
            down_ratio=msblock_down_ratio,
            mid_expand_ratio=msblock_mid_expand_ratio,
            attention_cfg=msblock_attention_cfg,
            channel_split_ratios=msblock_channel_split_ratios,
            layers_num=round(msblock_layers_num * deepen_factor),
            drop_layer=drop_layer,
            start_branch_id=msblock_start_branch_id,
            conv_group=conv_group,
            norm_cfg=norm_cfg,
            act_cfg=msblock_act_cfg,
            conv_cfg=conv_cfg,
        )
        super().__init__(
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            norm_cfg=norm_cfg,
            **kwargs,
        )

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """Build the top-down layer.

        Args:
            idx (int): Layer index.

        Returns:
            nn.Module: The top-down layer.
        """
        self.layer_config["kernel_sizes"] = self.kernel_sizes["top_down"][idx - 1]
        return MSBlock(
            make_divisible(
                (self.in_channels[idx - 1] + self.in_channels[idx]), self.widen_factor
            ),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            **self.layer_config,
        )

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """Build the bottom-up layer.

        Args:
            idx (int): Layer index.

        Returns:
            nn.Module: The bottom-up layer.
        """
        self.layer_config["kernel_sizes"] = self.kernel_sizes["bottom_up"][idx]
        return MSBlock(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]), self.widen_factor
            ),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            **self.layer_config,
        )
