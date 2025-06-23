_base_ = 'mmyolo::rtmdet/rotated/rtmdet-r_l_syncbn_fast_2xb4-36e_dota.py'

train_batch_size_per_gpu = 1
val_batch_size_per_gpu = 2

out_channels = 240
widen_factor = 0.42
deepen_factor = 1 / 3
mid_expand_ratio = 2
layers_num = 3
num_classes = 15

in_channels = [320, 640, 1280]
mid_channels = [160, 320, 640]

norm_cfg = dict(type='BN')
act_cfg = dict(type='SiLU', inplace=True)
in_down_ratio = 0.5

model = dict(
    backbone=dict(
        _delete_=True,
        _scope_='mmyolo',
        type='YOLOMS',
        arch='C3-K3579-ba234',
        layer_type='MSBlock_kxk_1x1_Layer',
        deepen_factor=deepen_factor,
        query_size=4,
        n=3,
        widen_factor=widen_factor,
        mid_expand_ratio=mid_expand_ratio,
        layers_num=layers_num,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg
    ),
    neck=dict(
        _delete_=True,
        _scope_='mmyolo',
        type='YOLOMSPAFPN',
        layer_type='MSBlock_kxk_1x1_Layer',
        kernel_sizes=dict(
            top_down=[(1, 3, 3), (1, 5, 5)],
            bottom_up=[(1, 5, 5), (1, 7, 7)]
        ),
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=in_channels,
        mid_channels=mid_channels,
        out_channels=out_channels,
        mid_expand_ratio=mid_expand_ratio,
        layers_num=layers_num,
        in_down_ratio=in_down_ratio,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg
    ),
    bbox_head=dict(
        head_module=dict(
            type='RTMDetRotatedSepBNHeadModule',
            num_classes=num_classes,
            widen_factor=widen_factor,
            in_channels=out_channels,
            feat_channels=out_channels
        )
    )
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(_delete_=True, type='yolov5_collate'),
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True)
)

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=False)
)

val_evaluator = dict(
    _delete_=True,
    type='mmrotate.DOTAMetric',
    metric='mAP',
    iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
)
test_evaluator = val_evaluator