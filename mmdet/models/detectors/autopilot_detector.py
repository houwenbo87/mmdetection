import torch.nn as nn

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class AutopilotDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 vehicle_head=None,
                 parkspace_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(AutopilotDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)
        else:
            self.bbox_head = None
        if vehicle_head is not None:
            vehicle_head.update(train_cfg=train_cfg)
            vehicle_head.update(test_cfg=test_cfg)
            self.vehicle_head = build_head(vehicle_head)
        else:
            self.vehicle_head = None
        if parkspace_head is not None:
            parkspace_head.update(train_cfg=train_cfg)
            parkspace_head.update(test_cfg=test_cfg)
            self.parkspace_head = build_head(parkspace_head)
        else:
            self.parkspace_head = None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(AutopilotDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.bbox_head:
            self.bbox_head.init_weights()
        if self.vehicle_head:
            self.vehicle_head.init_weights()
        if self.parkspace_head:
            self.parkspace_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_vehicles=None,
                      gt_vehicle_labels=None,
                      gt_vehicle_labels_mask=None,
                      gt_parkspaces=None,
                      gt_parkspace_labels=None,
                      ):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        losses = dict()
        if self.bbox_head is not None and gt_bboxes is not None:
            losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore)
            #outs = self.bbox_head(x)
            #loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            #losses = self.bbox_head.loss(
            #    *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if self.vehicle_head is not None and gt_vehicles is not None:
            vehicle_outs = self.vehicle_head(x)
            vehicle_loss_inputs = vehicle_outs + (gt_vehicles, gt_vehicle_labels, \
                                    gt_vehicle_labels_mask, img_metas, self.train_cfg)
            vehicle_loss = self.vehicle_head.loss(*vehicle_loss_inputs)
            losses.update(vehicle_loss)

        if self.parkspace_head is not None and gt_parkspaces is not None:
            parkspace_outs = self.parkspace_head(x)
            parkspace_loss_inputs = parkspace_outs + (gt_parkspaces, gt_parkspace_labels, img_metas, self.train_cfg)
            parkspace_loss = self.parkspace_head.loss(*parkspace_loss_inputs)
            losses.update(parkspace_loss)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        if self.vehicle_head is not None:
            vehicle_outs = self.vehicle_head(x)
            vehicle_list = self.vehicle_head.get_vehicles(
                *vehicle_outs, bbox_results[0], img_metas, rescale=rescale)
        else:
            vehicle_list = []
        
        return bbox_results[0]

    def simple_test_all(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        if self.vehicle_head is not None:
            vehicle_outs = self.vehicle_head(x)
            vehicle_list = self.vehicle_head.get_vehicles(
                *vehicle_outs, bbox_results[0], img_metas, rescale=rescale)
        else:
            vehicle_list = []
        
        return bbox_results[0], vehicle_list

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    @property
    def with_bbox(self):
        return (hasattr(self, 'bbox_head') and self.bbox_head is not None)

    @property
    def with_mask(self):
        return False

    @property
    def with_shared_head(self):
        return False