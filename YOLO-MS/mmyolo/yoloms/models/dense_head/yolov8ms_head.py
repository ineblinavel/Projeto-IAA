from torch import nn
from mmcv.cnn import ConvModule
from mmyolo.models.dense_heads.yolov8_head import YOLOv8HeadModule
from mmyolo.registry import MODELS
from ..utils import autopad
import torch


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
class YOLOv8MSHeadModule(YOLOv8HeadModule):
    """YOLOv8 Head Module with Multi-Scale Support."""

    def __init__(self,
                 reg_kernel_sizes=[[3], [5], [7]],
                 cls_kernel_sizes=[[3], [5], [7]],
                 groups=[4, 4, 4],
                 **kwargs):
        self.reg_kernel_sizes = reg_kernel_sizes
        self.cls_kernel_sizes = cls_kernel_sizes
        self.groups = groups
        super().__init__(**kwargs)

    def _init_layers(self):
        """Initialize convolutional layers in YOLOv8 head."""
        # Initialize decoupled head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        reg_out_channels = max(16, self.in_channels[0] // 4, self.reg_max * 4)
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            # Classification predictions
            cls_preds = []
            for j, cls_kernel_size in enumerate(self.cls_kernel_sizes[i]):
                in_channel = self.in_channels[i] if j == 0 else cls_out_channels
                cls_preds.append(
                    DWConv(
                        in_channel,
                        cls_out_channels,
                        cls_kernel_size,
                        groups=self.groups[i],
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    )
                )
            cls_preds.append(
                nn.Conv2d(
                    in_channels=cls_out_channels,
                    out_channels=self.num_classes,
                    kernel_size=1
                )
            )
            self.cls_preds.append(nn.Sequential(*cls_preds))

            # Regression predictions
            reg_preds = []
            for j, reg_kernel_size in enumerate(self.reg_kernel_sizes[i]):
                in_channel = self.in_channels[i] if j == 0 else reg_out_channels
                reg_preds.append(
                    DWConv(
                        in_channel,
                        reg_out_channels,
                        reg_kernel_size,
                        groups=self.groups[i],
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    )
                )
            reg_preds.append(
                nn.Conv2d(
                    in_channels=reg_out_channels,
                    out_channels=4 * self.reg_max,
                    kernel_size=1
                )
            )
            self.reg_preds.append(nn.Sequential(*reg_preds))

        # Register projection buffer
        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)
