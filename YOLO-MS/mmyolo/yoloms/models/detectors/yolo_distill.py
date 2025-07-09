from pathlib import Path
from typing import Union
from torch import Tensor

import torch
import torch.nn as nn


from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, reduce_mean
from mmyolo.models.detectors.yolo_detector import YOLODetector
from mmyolo.registry import MODELS
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.structures import SampleList

from mmyolo.models.utils import gt_instances_preprocess

from mmdet.structures.bbox import distance2bbox

from icecream import ic

@MODELS.register_module()
class YOLODistillDetector(YOLODetector):
    def __init__(self,
                 teacher_config: ConfigType,
                 teacher_ckpt: Union[str, Path],
                 kd_cfg: OptConfigType = None,
                 weight_transfer = False,
                 eval_same_layer = False,
                 **kwargs):
        super().__init__(**kwargs)
        # Build teacher model
        if isinstance(teacher_config, (str, Path)):
            teacher_config = Config.fromfile(teacher_config)
        if weight_transfer:
            load_checkpoint(self, teacher_ckpt, map_location="cpu", strict = False)

        self.teacher: nn.Module = MODELS.build(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(self.teacher, teacher_ckpt, map_location='cpu', strict = True)
        # In order to reforward teacher model,
        # set requires_grad of teacher model to False
        self.freeze(self.teacher)
        if kd_cfg.get('loss_backbone_kd', None):
            self.loss_backbone_kd = MODELS.build(kd_cfg['loss_backbone_kd'])
        if kd_cfg.get('loss_neck_kd', None):
            self.loss_neck_kd = MODELS.build(kd_cfg['loss_neck_kd'])
        if kd_cfg.get('loss_cls_kd', None):
            self.loss_cls_kd = MODELS.build(kd_cfg['loss_cls_kd'])
        if kd_cfg.get('loss_bbox_kd', None):
            self.loss_bbox_kd = MODELS.build(kd_cfg['loss_bbox_kd'])

        
        
        if eval_same_layer:
            for name, param in self.named_parameters():
                if "teacher" in name:
                    continue
                if name in self.teacher.state_dict():
                    teacher_size = self.teacher.state_dict()[name].size()
                    size = param.size()
                    if teacher_size == size:
                        param.requires_grad = False
                        continue
                ic("Training {}".format(name))
                
    
    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = batch_inputs
        teacher_x = batch_inputs

        losses = dict()

        outs = []
        teacher_outs = []
        for i, layer_name in enumerate(self.backbone.layers):
            layer = getattr(self.backbone, layer_name)
            teacher_layer = getattr(self.teacher.backbone, layer_name)
            
            x = layer(x)
            teacher_x = teacher_layer(teacher_x)
            if i in self.backbone.out_indices:    
                if hasattr(self, "loss_backbone_kd"):
                    with torch.cuda.amp.autocast(enabled=False):
                        losses["loss_backbone_kd_{}".format(i)] = self.loss_backbone_kd(x, 
                                                                                        teacher_x.detach())
                outs.append(x)
                teacher_outs.append(teacher_x)
                        
        outs = self.neck(outs)
        teacher_outs = self.teacher.neck(teacher_outs)

        for i, (x, teacher_x) in enumerate(zip(outs, teacher_outs)):
            if hasattr(self, "loss_neck_kd"):
                with torch.cuda.amp.autocast(enabled=False):
                    losses["loss_neck_kd_{}".format(i)] = self.loss_neck_kd(x, teacher_x.detach())

        losses.update(self.kd_loss_by_feats(outs, teacher_outs, batch_data_samples))
        return losses
    

    def kd_loss_by_feats(self, x, teacher_x,  batch_data_samples):
        cls_scores, bbox_preds = self.bbox_head(x)
        tea_cls_scores, tea_bbox_preds = self.teacher.bbox_head(teacher_x)

        batch_gt_instances = batch_data_samples['bboxes_labels']
        batch_img_metas = batch_data_samples['img_metas']

        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.bbox_head.prior_generator.num_levels

        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy

        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
        device = cls_scores[0].device

        if featmap_sizes != self.bbox_head.featmap_sizes_train:
            self.bbox_head.featmap_sizes_train = featmap_sizes
            mlvl_priors_with_stride = self.bbox_head.prior_generator.grid_priors(featmap_sizes,
                                                                                 device=device, 
                                                                                 with_stride=True)
            self.bbox_head.flatten_priors_train = torch.cat(mlvl_priors_with_stride, dim=0)
            
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.bbox_head.cls_out_channels)
            for cls_score in cls_scores
        ], 1).contiguous()

        flatten_tea_cls_scores = torch.cat([
            tea_cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, 
                                                      self.bbox_head.cls_out_channels)
            for tea_cls_score in tea_cls_scores
        ], 1).contiguous()

        flatten_bboxes = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ], 1)
        flatten_bboxes = flatten_bboxes * self.bbox_head.flatten_priors_train[..., -1,
                                                                    None]
        flatten_bboxes = distance2bbox(self.bbox_head.flatten_priors_train[..., :2],
                                       flatten_bboxes)
        

        flatten_tea_bboxes = torch.cat([
            tea_bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for tea_bbox_pred in tea_bbox_preds
        ], 1)
        flatten_tea_bboxes = flatten_tea_bboxes * self.bbox_head.flatten_priors_train[..., -1,
                                                                                      None]
        flatten_tea_bboxes = distance2bbox(self.bbox_head.flatten_priors_train[..., :2],
                                           flatten_tea_bboxes)
        

        assigned_result = self.bbox_head.assigner(flatten_bboxes.detach(),
                                                  flatten_cls_scores.detach(),
                                                  self.bbox_head.flatten_priors_train, 
                                                  gt_labels,
                                                  gt_bboxes, 
                                                  pad_bbox_flag)
        
        labels = assigned_result['assigned_labels'].reshape(-1)
        label_weights = assigned_result['assigned_labels_weights'].reshape(-1)
        bbox_targets = assigned_result['assigned_bboxes'].reshape(-1, 4)
        assign_metrics = assigned_result['assign_metrics'].reshape(-1)
        cls_preds = flatten_cls_scores.reshape(-1, self.bbox_head.num_classes)
        bbox_preds = flatten_bboxes.reshape(-1, 4)

        tea_cls_preds = flatten_tea_cls_scores.reshape(-1, self.bbox_head.num_classes)
        tea_bbox_preds = flatten_tea_bboxes.reshape(-1, 4)


        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.bbox_head.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        avg_factor = reduce_mean(assign_metrics.sum()).clamp_(min=1).item()

        loss_cls = self.bbox_head.loss_cls(
            cls_preds, (labels, assign_metrics),
            label_weights,
            avg_factor=avg_factor)

        if len(pos_inds) > 0:
            loss_bbox = self.bbox_head.loss_bbox(
                bbox_preds[pos_inds],
                bbox_targets[pos_inds],
                weight=assign_metrics[pos_inds],
                avg_factor=avg_factor)
        else:
            loss_bbox = bbox_preds.sum() * 0

        losses = dict(loss_cls=loss_cls, 
                      loss_bbox=loss_bbox)
        
        with torch.cuda.amp.autocast(enabled=False):
            if hasattr(self, "loss_cls_kd"):
                loss_cls_kd = self.loss_cls_kd(cls_preds, 
                                               tea_cls_preds.detach(),
                                               label_weights,
                                               avg_factor=avg_factor)
                losses.update({"loss_cls_kd": loss_cls_kd})
            if hasattr(self, "loss_bbox_kd"):
                if len(pos_inds) > 0:
                    loss_bbox_kd = self.loss_bbox_kd(bbox_preds[pos_inds],
                                                    tea_bbox_preds[pos_inds],
                                                    weight=assign_metrics[pos_inds],
                                                    avg_factor=avg_factor)
                else:
                    loss_bbox_kd = bbox_preds.sum() * 0
                losses.update({"loss_bbox_kd": loss_bbox_kd})
        return losses

