_base_ = '../yoloms-xs_syncbn_fast_8xb32-300e_coco.py'

load_from = 'ckpts/yoloms-xs_syncbn_fast_8xb32-300e_coco.pth'

data_root = 'data/RTTS/'
class_name = ('bicycle', 'bus', 'car', 'motorbike', 'person')
palette = [(255, 97, 0), (0, 201, 87), (176, 23, 31), (138, 43, 226),
           (30, 144, 255)]

num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=palette)

num_epochs_stage2 = 5

max_epochs = 100
train_batch_size_per_gpu = 8
train_num_workers = 4
val_batch_size_per_gpu = 1
val_num_workers = 2

lr = 0.001


model = dict(
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))

train_dataloader = dict(
    _delete_=True,
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type='yolov5_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type="YOLOv5CocoDataset",
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations_json/rtts_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    _delete_=True,
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type="YOLOv5CocoDataset",
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations_json/rtts_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        batch_shapes_cfg=_base_.batch_shapes_cfg,
        pipeline=_base_.test_pipeline))

test_dataloader = val_dataloader

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=_base_.lr_start_factor,
        by_epoch=False,
        begin=0,
        end=30),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

_base_.custom_hooks[1].switch_epoch = max_epochs - num_epochs_stage2

val_evaluator = dict(ann_file=data_root + 'annotations_json/rtts_val.json')
test_evaluator = val_evaluator


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epochs, val_interval=10)