# -*- coding: utf-8 -*-

import torch
import numpy as np
from typing import Dict
from torch import nn
# from typing import Any, Iterator, List, Union
# import pycocotools.mask as mask_utils
# from torch.nn import functional as F
from detectron2.layers import Conv2d, ShapeSpec, get_norm
# import fvcore.nn.weight_init as weight_init
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler, MultiROIPooler
# import cv2
import fvcore.nn.weight_init as weight_init
from torch.nn import functional as F
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals, select_proposals_with_visible_keypoints
from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_inference,keypoint_rcnn_loss
from .densepose_head import (
    build_densepose_data_filter,
    build_densepose_head,
    build_densepose_losses,
    build_densepose_predictor,
    densepose_inference,
    DensePoseInterLosses,
    dp_keypoint_rcnn_loss,
    DensePoseKeypointsPredictor,
    DensePoseDataFilter,
)
from .semantic_mask_head import (
    build_semantic_mask_data_filter,
    DpSemSegFPNHead
)
class MultiInstanceDecoder(torch.nn.Module):

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec], in_features):
        super(MultiInstanceDecoder, self).__init__()

        # fmt: off
        self.in_features      = in_features
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        num_out_dims           = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_OUT_DIMS
        conv_dims             = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_CONV_DIMS
        self.common_stride    = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_COMMON_STRIDE
        norm                  = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NORM
        # fmt: on

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, conv_dims),
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = Conv2d(conv_dims, num_out_dims, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    def forward(self, features):
        for i, _ in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[i])
            else:
                x = x + self.scale_heads[i](features[i])
        x = self.predictor(x)
        return x
@ROI_HEADS_REGISTRY.register()
class DensePoseROIHeads(StandardROIHeads):
    """
    A Standard ROIHeads which contains an addition of DensePose head.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_densepose_head(cfg, input_shape)
        self._init_dpsemseg_head(cfg)
        self._init_dp_keypoint_head(cfg,input_shape)


    def _init_dpsemseg_head(self, cfg):
        self.dp_semseg_on = cfg.MODEL.ROI_DENSEPOSE_HEAD.SEMSEG_ON

        if not self.dp_semseg_on:
            return
        input_shape = {
            name: ShapeSpec(
                channels=self.feature_channels[name], stride=self.feature_strides[name]
            )
            for name in self.in_features
        }
        self.common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        self.dp_semseg_head = DpSemSegFPNHead(cfg, input_shape)
        self.sem_mask_data_filter = build_semantic_mask_data_filter(cfg)
    def _init_dp_keypoint_head(self, cfg,input_shape):
        # fmt: off

        self.dp_keypoint_on                         = cfg.MODEL.ROI_DENSEPOSE_HEAD.KPT_ON
        if not self.dp_keypoint_on:
            return
        self.normalize_loss_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.keypoint_loss_weight                = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
        dp_pooler_resolution = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
        dp_pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        dp_pooler_sampling_ratio = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO
        dp_pooler_type = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE
        in_channels = [self.feature_channels[f] for f in self.in_features][0]
        if not self.densepose_on:
            self.use_mid = cfg.MODEL.ROI_DENSEPOSE_HEAD.MID_ON
            if self.use_mid:
                self.mid_decoder = MultiInstanceDecoder(cfg, input_shape, self.in_features)
                dp_pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)

            self.densepose_pooler = ROIPooler(
                output_size=dp_pooler_resolution,
                scales=dp_pooler_scales,
                sampling_ratio=dp_pooler_sampling_ratio,
                pooler_type=dp_pooler_type,
            )
            self.densepose_head = build_densepose_head(cfg, in_channels)
            self.keypoint_predictor = DensePoseKeypointsPredictor(cfg, self.densepose_head.n_out_channels)

    def _init_densepose_head(self, cfg, input_shape):
        # fmt: off
        self.cfg                   = cfg
        self.densepose_on          = cfg.MODEL.DENSEPOSE_ON and cfg.MODEL.ROI_DENSEPOSE_HEAD.RCNN_HEAD_ON
        if not self.densepose_on:
            return
        self.densepose_data_filter = build_densepose_data_filter(cfg)
        dp_pooler_resolution       = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
        # dp_multi_pooler_res        = ((28,28),(14,14),(14,14),(7,7))
        dp_pooler_scales           = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        dp_pooler_sampling_ratio   = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO
        dp_pooler_type             = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE
        self.use_mid = cfg.MODEL.ROI_DENSEPOSE_HEAD.MID_ON
        self.mask_thresh = cfg.MODEL.ROI_DENSEPOSE_HEAD.FG_MASK_THRESHOLD
        if self.use_mid:
            self.mid_decoder = MultiInstanceDecoder(cfg, input_shape, self.in_features)
            dp_pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        self.inter_super_on = False
        self.inter_weight          = cfg.MODEL.ROI_DENSEPOSE_HEAD.INTER_WEIGHTS
        if cfg.MODEL.ROI_DENSEPOSE_HEAD.NAME == 'DensePosePIDHead':
            self.inter_super_on = True

        # fmt: on
        in_channels = [self.feature_channels[f] for f in self.in_features][0]
        self.densepose_pooler = ROIPooler(
            output_size=dp_pooler_resolution,
            scales=dp_pooler_scales,
            sampling_ratio=dp_pooler_sampling_ratio,
            pooler_type=dp_pooler_type,
        )
        self.densepose_head = build_densepose_head(cfg, in_channels)
        self.densepose_predictor = build_densepose_predictor(
            cfg, self.densepose_head.n_out_channels
        )
        self.densepose_losses = build_densepose_losses(cfg)
        self.densepose_inter_losses = DensePoseInterLosses(cfg)

    def _forward_semsegs(self, features, instances, extra):
        if not self.dp_semseg_on:
            return
        if self.training:
            im_h, im_w = int(features[0].size(2)* self.common_stride), \
                         int(features[0].size(3)* self.common_stride)
            # proposals_with_targets, _ = select_foreground_proposals(instances, self.num_classes)
            # gt_sem_seg = self.sem_mask_data_filter(proposals_with_targets, im_h, im_w)
            gt_sem_seg = self.sem_mask_data_filter(extra, im_h, im_w)
            sem_seg_results, sem_seg_losses, latent_features = self.dp_semseg_head(features, gt_sem_seg)
            # sem_seg_results = gt_sem_seg.float().unsqueeze(1)
        else:
            gt_sem_seg = None
            sem_seg_results, sem_seg_losses, latent_features = self.dp_semseg_head(features, gt_sem_seg)

        return sem_seg_results, sem_seg_losses, latent_features

    def _forward_dp_keypoint(self, keypoint_logits, instances):

        if not self.dp_keypoint_on:
            return {} if self.training else instances
        num_images = len(instances)
        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            normalizer = (
                num_images
                * self.batch_size_per_image
                * self.positive_sample_fraction
                * keypoint_logits.shape[1]
            )
            loss = dp_keypoint_rcnn_loss(
                keypoint_logits,
                instances,
                normalizer=None if self.normalize_loss_by_visible_keypoints else normalizer,
            )
            return {"loss_keypoint": loss * self.keypoint_loss_weight}
        else:
            keypoint_rcnn_inference(keypoint_logits, instances)
            return instances

    def forward_dp_keypoint(self, features, instances):

        if not self.dp_keypoint_on:
            return {} if self.training else instances
        num_images = len(instances)
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            # proposals = self.keypoint_data_filter(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            # print('after:',len(proposal_boxes))
            # if len(proposal_boxes) == 0:
            #     return {"loss_keypoint": 0.}
            # if len(proposal_boxes) > 0:
            if self.use_mid:
                features = [self.mid_decoder(features)]
            keypoint_features = self.densepose_pooler(features, proposal_boxes)
            keypoint_output = self.densepose_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_output)
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            normalizer = (
                num_images
                * self.batch_size_per_image
                * self.positive_sample_fraction
                * keypoint_logits.shape[1]
            )
            loss = keypoint_rcnn_loss(
                keypoint_logits,
                proposals,
                normalizer=None if self.normalize_loss_by_visible_keypoints else normalizer,
            )
            return {"loss_keypoint": loss * self.keypoint_loss_weight}
        else:
            if self.use_mid:
                features = [self.mid_decoder(features)]
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.densepose_pooler(features, pred_boxes)
            keypoint_output = self.densepose_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_output)
            keypoint_rcnn_inference(keypoint_logits, instances)
            return instances

    def _forward_densepose(self, features, instances):
        """
        Forward logic of the densepose prediction branch.

        Args:
            features (list[Tensor]): #level input features for densepose prediction
            instances (list[Instances]): the per-image instances to train/predict densepose.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "densepose" and return it.
        """
        if not self.densepose_on:
            return {} if self.training else instances

        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals_dp = self.densepose_data_filter(proposals)
            if len(proposals_dp) > 0:
                proposal_boxes = [x.proposal_boxes for x in proposals_dp]
                if self.use_mid:
                    features = [self.mid_decoder(features)]
                features_dp = self.densepose_pooler(features, proposal_boxes)
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_outputs, _ = self.densepose_predictor(densepose_head_outputs)
                if self.dp_keypoint_on:
                    keypoints_output = densepose_outputs[-1]
                    densepose_outputs = densepose_outputs[:-1]
                densepose_loss_dict = self.densepose_losses(proposals_dp, densepose_outputs)
                if self.dp_keypoint_on:
                    kpt_loss_dict = self._forward_dp_keypoint(keypoints_output, proposals_dp)
                    for _, k in enumerate(kpt_loss_dict.keys()):
                        densepose_loss_dict[k] = kpt_loss_dict[k]
                return densepose_loss_dict
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            if self.use_mid:
                features = [self.mid_decoder(features)]
            features_dp = self.densepose_pooler(features, pred_boxes)
            if len(features_dp) > 0:
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_outputs, _ = self.densepose_predictor(densepose_head_outputs)
                if self.dp_keypoint_on:
                    keypoints_output = densepose_outputs[-1]
                    densepose_outputs = densepose_outputs[:-1]
                    instances = self._forward_dp_keypoint(keypoints_output, instances)
            else:
                # If no detection occured instances
                # set densepose_outputs to empty tensors
                empty_tensor = torch.zeros(size=(0, 0, 0, 0), device=features_dp.device)
                densepose_outputs = tuple([empty_tensor] * 5)

            densepose_inference(densepose_outputs, instances, self.mask_thresh)
            return instances

    def forward(self, images, features, proposals, targets=None, extra=None):
        features_list = [features[f] for f in self.in_features]

        instances, losses = super().forward(images, features, proposals, targets)

        del targets, images

        if self.training:
            losses.update(self._forward_densepose(features_list, instances))
        else:
            instances = self._forward_densepose(features_list, instances)
        return instances, losses


def select_proposals_idx_with_visible_keypoints(keypoints_logits, proposals):
    start = 0
    selected_logits = []
    selected_proposals = []
    for proposals_per_image in proposals:
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0])
            & (xs <= proposal_boxes[:, :, 2])
            & (ys >= proposal_boxes[:, :, 1])
            & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = torch.nonzero(selection).squeeze(1)
        selected_proposals.append(proposals_per_image[selection_idxs])
        selection_idxs += start
        selected_logits_per_img = keypoints_logits[selection_idxs]
        start = start + len(selected_logits_per_img)
        selected_logits.append(selected_logits_per_img)

    selected_logits = torch.cat(selected_logits, 0)
    return selected_logits, selected_proposals