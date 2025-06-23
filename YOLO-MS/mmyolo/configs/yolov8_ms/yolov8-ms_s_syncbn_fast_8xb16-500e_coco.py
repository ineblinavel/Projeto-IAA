# Reference to
# https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py
_base_ = './yolov8-ms_n_syncbn_fast_8xb16-500e_coco.py'

train_batch_size_per_gpu = 16

msblock_layers_num = 3
deepen_factor = 1 / 3
widen_factor = 0.35

model = dict(
    backbone=dict(deepen_factor=deepen_factor,
                  widen_factor=widen_factor,
                  msblock_layers_num=msblock_layers_num),
    neck=dict(deepen_factor=deepen_factor,
              widen_factor=widen_factor,
              msblock_layers_num=msblock_layers_num),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(_delete_=True, type='yolov5_collate'),
                        sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True))

auto_scale_lr = dict(enable=True, base_batch_size=16 * 8)
