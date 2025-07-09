_base_ = 'mmdet::rtmdet/rtmdet-ins_l_8xb32-300e_coco.py'

# Number of classes for classification
num_classes = 80
# Batch size of a single GPU during training
train_batch_size_per_gpu = 16
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 5
# persistent_workers must be False if num_workers is 0.
persistent_workers = True

# Batch size of a single GPU during validation
val_batch_size_per_gpu = 5
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 5

# The scaling factor that controls the depth of the network structure
deepen_factor = 1 / 3
# The scaling factor that controls the width of the network structure
widen_factor = 0.42

# Input channels of PAFPN
in_channels = [320, 640, 1280]
# Middle channels of PAFPN
mid_channels = [160, 320, 640]
# Output channels of PAFPN
out_channels = 240

# The type of layer in MS-Block
msblock_layer_type = "MSBlock_kxk_1x1_Layer"
# The downsample ratio of MS-Block in Backbone
backbone_msblock_down_ratio = 1
# The downsample ratio of MS-Block in PAFPN
neck_msblock_down_ratio = 0.5
# Channel expand ratio for each branch in MS-Block
msblock_mid_expand_ratio = 2
# Channel down ratio for downsample conv layer in MS-Block
msblock_layers_num = 3
# The split ratio of MS-Block
msblock_channel_split_ratios = [1, 1, 1]
# The start branch id of MS-Block
msblock_start_branch_id = 2

# Normalization config
norm_cfg = dict(type='BN')
# Activation config
act_cfg = dict(type='SiLU', inplace=True)

# Kernel sizes of MS-Block in PAFPN
kernel_sizes = dict(
    top_down=[(1, 3, 3), (1, 5, 5)],
    bottom_up=[(1, 5, 5), (1, 7, 7)]
)

model = dict(
    backbone=dict(
        _delete_=True,
        _scope_='mmyolo',
        type='YOLOMS',
        arch="C3-K3579-A234",
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        norm_eval=False,
        act_cfg=act_cfg,
        msblock_layer_type = msblock_layer_type,
        msblock_mid_expand_ratio=msblock_mid_expand_ratio,
        msblock_layers_num=msblock_layers_num,
        msblock_down_ratio=backbone_msblock_down_ratio,
        msblock_attention_cfg=dict(
            type="GQL",
            length=3,
            size=4
        ),
        msblock_start_branch_id=msblock_start_branch_id,
        msblock_norm_cfg=norm_cfg,
        msblock_act_cfg=act_cfg),
    neck=dict(
        _delete_=True,
        _scope_='mmyolo',
        type='YOLOMSPAFPN',
        in_channels=in_channels,
        mid_channels=mid_channels,
        out_channels=out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        kernel_sizes=kernel_sizes,

        msblock_layer_type = msblock_layer_type,
        msblock_down_ratio=neck_msblock_down_ratio,
        msblock_mid_expand_ratio=msblock_mid_expand_ratio,
        msblock_layers_num=msblock_layers_num,
        msblock_channel_split_ratios = msblock_channel_split_ratios,
        msblock_start_branch_id=msblock_start_branch_id,
        msblock_attention_cfg=None,
        msblock_act_cfg=act_cfg,

        norm_cfg=norm_cfg,
        act_cfg=act_cfg),
    bbox_head=dict(
        in_channels=100,
        feat_channels=100,
    )
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
)

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu, num_workers=val_num_workers)