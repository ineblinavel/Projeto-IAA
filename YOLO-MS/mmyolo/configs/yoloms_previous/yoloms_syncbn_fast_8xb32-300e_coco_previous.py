# Reference:
# https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py
_base_ = 'mmyolo::rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py'

# ======================== Frequently Modified Parameters ========================
# ----- Data Related -----
data_root = 'data/coco/'
train_ann_file = 'annotations/instances_train2017.json'  # Train annotation file
train_data_prefix = 'train2017/'  # Train image path prefix
val_ann_file = 'annotations/instances_val2017.json'  # Validation annotation file
val_data_prefix = 'val2017/'  # Validation image path prefix

num_classes = 80  # Number of classes
train_batch_size_per_gpu = 32  # Batch size per GPU during training
train_num_workers = 5  # Number of workers for training data loading
val_num_workers = 5  # Number of workers for validation data loading
persistent_workers = True  # Persistent workers for data loading

# ======================== Possible Modified Parameters =========================
# ----- Data Related -----
val_batch_size_per_gpu = 32  # Batch size per GPU during validation

# ----- Model Architecture -----
layers_num = 3  # Number of layers in MS-Block
deepen_factor = 2 / 3  # Depth scaling factor
widen_factor = 0.8  # Width scaling factor

# PAFPN Channels
in_channels = [320, 640, 1280]  # Input channels
mid_channels = [160, 320, 640]  # Middle channels
out_channels = 240  # Output channels

# MS-Block Configurations
msblock_layer_type = "MSBlockBottleNeckLayer"
backbone_msblock_down_ratio = 1  # Downsample ratio in Backbone
neck_msblock_down_ratio = 0.5  # Downsample ratio in PAFPN
msblock_mid_expand_ratio = 2  # Channel expand ratio for each branch
msblock_layers_num = 3  # Number of layers in MS-Block
msblock_channel_split_ratios = [1, 1, 1]  # Channel split ratios

# Normalization and Activation Configurations
norm_cfg = dict(type='BN')  # Normalization config
act_cfg = dict(type='SiLU', inplace=True)  # Activation config

# Kernel Sizes for MS-Block in PAFPN
kernel_sizes = dict(
    bottom_up=[[1, (3, 3), (3, 3)], [1, (3, 3), (3, 3)]],
    top_down=[[1, (3, 3), (3, 3)], [1, (3, 3), (3, 3)]]
)

loss_bbox_weight = 2.0  # Bounding box loss weight

# ======================== Unmodified in Most Cases =========================
# Model Configuration
model = dict(
    backbone=dict(
        _delete_=True,
        type='YOLOMS',
        arch='C3-K3579',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        norm_eval=False,
        act_cfg=act_cfg,
        msblock_layer_type=msblock_layer_type,
        msblock_down_ratio=backbone_msblock_down_ratio,
        msblock_mid_expand_ratio=msblock_mid_expand_ratio,
        msblock_layers_num=msblock_layers_num,
        msblock_norm_cfg=norm_cfg,
        msblock_act_cfg=act_cfg
    ),
    neck=dict(
        _delete_=True,
        type='YOLOMSPAFPN',
        in_channels=in_channels,
        mid_channels=mid_channels,
        out_channels=out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        kernel_sizes=kernel_sizes,
        msblock_layer_type=msblock_layer_type,
        msblock_down_ratio=neck_msblock_down_ratio,
        msblock_mid_expand_ratio=msblock_mid_expand_ratio,
        msblock_layers_num=msblock_layers_num,
        msblock_channel_split_ratios=msblock_channel_split_ratios,
        msblock_act_cfg=act_cfg,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg
    ),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            in_channels=out_channels,
            feat_channels=out_channels,
            num_classes=num_classes,
            act_cfg=dict(inplace=True, type='LeakyReLU')
        ),
        loss_bbox=dict(
            type='mmdet.DIoULoss',
            loss_weight=loss_bbox_weight
        )
    ),
    train_cfg=dict(
        assigner=dict(num_classes=num_classes)
    )
)

# Dataloader Configurations
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    collate_fn=dict(_delete_=True, type='yolov5_collate'),
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True),
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)
    )
)

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        test_mode=True
    )
)

test_dataloader = val_dataloader

# Auto-scaling Learning Rate
auto_scale_lr = dict(enable=True, base_batch_size=32 * 8)
