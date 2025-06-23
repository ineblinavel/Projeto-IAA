_base_ = './yoloms-s_syncbn_fast_8xb32-300e_coco_previous.py'

model = dict(backbone=dict(msblock_out_attention_cfg=dict(type='SE')))