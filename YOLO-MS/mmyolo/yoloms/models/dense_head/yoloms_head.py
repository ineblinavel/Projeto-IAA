from typing import Tuple
from torch import nn, Tensor
from mmyolo.models.dense_heads.rtmdet_head import RTMDetSepBNHeadModule
from mmyolo.registry import MODELS
from mmcv.cnn import ConvModule
from ..utils import autopad


class DWConv(nn.Module):
    """Depthwise Convolution Module."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 conv_cfg=None,
                 norm_cfg=None,
                 groups=4,
                 act_cfg=None):
        super().__init__()
        if kernel_size == 3:
            groups = 1
        else:
            groups = groups if groups != -1 else out_channels

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=autopad(kernel_size),
            groups=groups,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

    def forward(self, x):
        return self.conv(x)


@MODELS.register_module()
class YOLOMSHeadModule(RTMDetSepBNHeadModule):
    """YOLOMS Head Module with Multi-Scale Support."""

    def __init__(self,
                 reg_kernel_sizes=[[3], [5], [7]],
                 cls_kernel_sizes=[[3], [5], [7]],
                 groups=[4, 4, 4],
                 **kwargs):
        self.reg_kernel_sizes = reg_kernel_sizes
        self.cls_kernel_sizes = cls_kernel_sizes
        self.groups = groups
        super().__init__(**kwargs)

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        super().init_weights()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()

        for n in range(len(self.featmap_strides)):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()

            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    DWConv(
                        chn,
                        self.feat_channels,
                        self.cls_kernel_sizes[n][i],
                        groups=self.groups[n],
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    )
                )
                reg_convs.append(
                    DWConv(
                        chn,
                        self.feat_channels,
                        self.reg_kernel_sizes[n][i],
                        groups=self.groups[n],
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    )
                )

            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.num_classes,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2
                )
            )
            self.rtm_reg.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * 4,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2
                )
            )

        if self.share_conv:
            for n in range(len(self.featmap_strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
        """
        cls_scores = []
        bbox_preds = []

        for idx, x in enumerate(feats):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)
            reg_dist = self.rtm_reg[idx](reg_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)

        return tuple(cls_scores), tuple(bbox_preds)

