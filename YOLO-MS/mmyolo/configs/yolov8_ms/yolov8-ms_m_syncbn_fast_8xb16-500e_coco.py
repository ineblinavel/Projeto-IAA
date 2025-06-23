# Reference to
# https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py
_base_ = './yolov8-ms_n_syncbn_fast_8xb16-500e_coco.py'

msblock_layers_num = 3
deepen_factor = 1 / 3
widen_factor = 0.5
last_stage_out_channels = 1280

model = dict(
    backbone=dict(deepen_factor=deepen_factor,
                  widen_factor=widen_factor,
                  last_stage_out_channels=last_stage_out_channels,
                  
                  msblock_layer_type = "MSBlockBottleNeckLayer",
                  msblock_mid_expand_ratio = 2,
                  msblock_layers_num=msblock_layers_num),
    neck=dict(deepen_factor=deepen_factor,
              widen_factor=widen_factor,
              in_channels=[320, 640, last_stage_out_channels],
              out_channels=[320, 640, last_stage_out_channels],
              
              msblock_layer_type = "MSBlockBottleNeckLayer",
              msblock_mid_expand_ratio = 2,
              msblock_layers_num=msblock_layers_num),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor,
                                    in_channels=[320, 640, last_stage_out_channels])))


train_batch_size_per_gpu = 16
affine_scale = 0.9
mixup_prob = 0.1

img_scale = _base_.img_scale
pre_transform = _base_.pre_transform
last_transform = _base_.last_transform
auto_scale_lr = dict(enable=True, base_batch_size=16 * 8)

mosaic_affine_transform = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]

train_pipeline = [
    *pre_transform, *mosaic_affine_transform,
    dict(
        type='YOLOv5MixUp',
        prob=mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_transform]),
    *last_transform
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=100,
        border_val=(114, 114, 114)), *last_transform
]

train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(_delete_=True, type='yolov5_collate'),
                        sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True),
                        dataset=dict(pipeline=train_pipeline))
_base_.custom_hooks[1].switch_pipeline = train_pipeline_stage2


val_dataloader = dict(sampler=dict(_delete_=True, type='DefaultSampler', shuffle=False))