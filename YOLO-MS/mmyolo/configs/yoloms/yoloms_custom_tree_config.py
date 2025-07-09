_base_ = './yoloms-s_syncbn_fast_8xb32-300e_coco.py'

data_root = '/content/Projeto-IAA/dataset/'
class_name = ('tree',)
num_classes = len(class_name)
metainfo = dict(classes=class_name)

model = dict(
    bbox_head=dict(
        num_classes=num_classes
    )
)

train_dataloader = dict(
    dataset=dict(
        _delete_=True,  
        type='YOLOv5DetDataset', 
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.txt',
        data_prefix=dict(img='images/train/')
    )
)

# 5. Configuração do Dataloader de Validação
val_dataloader = dict(
    dataset=dict(
        type='YOLOv5DetDataset', 
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val.txt',
        data_prefix=dict(img='images/val/'),
        test_mode=True 
    )
)

test_dataloader = val_dataloader

# 6. Configuração do Avaliador de Validação
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json'
)
test_evaluator = val_evaluator

# 7. Hooks (opcional, sua configuração está correta)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10)
)