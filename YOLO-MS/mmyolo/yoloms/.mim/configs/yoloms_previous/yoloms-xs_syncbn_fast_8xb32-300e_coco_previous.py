_base_ = './yoloms-s_syncbn_fast_8xb32-300e_coco_previous.py'

deepen_factor = 1/3
widen_factor =  0.4

model = dict(backbone=dict(deepen_factor=deepen_factor,
                           widen_factor=widen_factor),
             neck=dict(deepen_factor=deepen_factor,
                       widen_factor=widen_factor),
             bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

default_hooks = dict(checkpoint=dict(interval=10, max_keep_ckpts=100, save_best='auto'),
                     logger=dict(type='LoggerHook', interval=5))
