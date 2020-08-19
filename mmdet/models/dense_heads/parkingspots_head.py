import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead

from ...utils import Debugger

import os
import cv2
import math
import numpy as np
import shutil

INF = 1e8

# 针对APA全景拼接图中的车位检测

@HEADS.register_module
class ParkingspotsHead(nn.Module):
    """
    Fully Convolutional One-Stage Object Detection head from [1]_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.

    References:
        .. [1] https://arxiv.org/abs/1904.01355

    Example:
        >>> self = ParkingspotsHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 num_keypts,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 in_feat_index=(0, 1, 2, 3, 4),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_hm=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_hm_kp=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_kps=dict(
                     type='SmoothL1Loss',
                     reduction='sum',
                     loss_weight=1.0),
                 loss_offset=dict(
                     type='SmoothL1Loss',
                     reduction='sum',
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 train_cfg=None,
                 test_cfg=None):
        super(ParkingspotsHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.in_feat_index = in_feat_index
        self.regress_ranges = regress_ranges

        self.loss_cls = build_loss(loss_cls)
        self.loss_hm = build_loss(loss_hm)
        self.loss_hm_kp = build_loss(loss_hm_kp)
        self.loss_kps = build_loss(loss_kps)
        self.loss_offset = build_loss(loss_offset)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self.num_keypts = num_keypts

        # cls 2, hm 1, hm_kp 4, kp_offset 2, kps 8, bbox 4
        # hm 中心点的热力图
        # hm_kp 关键点的热力图
        # kp_offset 每个关键点的偏移值
        # kps 每个关键点相对于中心点的偏移值
        # reg 中心点的偏移值, 变更为fcos的方式
        # bbox 每个点到四周的距离值
        self.heads = {'cls': self.cls_out_channels, 'hm' : 1, 'hm_kp' : self.num_keypts, 'kp_offset' : 2, 'kps' : self.num_keypts * 2, 'bbox' : 4}
        #self.out_channels = 1 + self.num_keypts + 2 + self.num_keypts*2 + 2 + 2

        self.iter = 0

        assert len(self.strides) == 1
        assert len(self.strides) == len(self.in_feat_index)

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.offset_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.offset_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.keypt_cls = nn.Conv2d(self.feat_channels, self.heads['cls'], 3, padding=1)
        self.keypt_hm = nn.Conv2d(self.feat_channels, self.heads['hm'], 3, padding=1)
        self.keypt_hm_kp = nn.Conv2d(self.feat_channels, self.heads['hm_kp'], 3, padding=1)
        self.keypt_kps = nn.Conv2d(self.feat_channels, self.heads['kps'], 3, padding=1)
        self.keypt_kp_offset = nn.Conv2d(self.feat_channels, self.heads['kp_offset'], 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        #self.extra_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.offset_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.keypt_cls, std=0.01, bias=bias_cls)
        normal_init(self.keypt_hm, std=0.01, bias=bias_cls)
        normal_init(self.keypt_hm_kp, std=0.01, bias=bias_cls)
        normal_init(self.keypt_kps, std=0.01)
        normal_init(self.keypt_kp_offset, std=0.01)

    def forward(self, feats):
        assert len(self.strides) == 1
        return self.forward_single(feats[self.in_feat_index[0]], self.cls_convs, self.reg_convs, self.offset_convs,
            self.keypt_cls, self.keypt_hm, self.keypt_hm_kp, self.keypt_kps, self.keypt_kp_offset,
            self.strides, self.scales[0])
        #return multi_apply(self.forward_single, feats, self.cls_convs, self.reg_convs,
        #    self.keypt_hm, self.keypt_hm_kp, self.keypt_kp_offset, self.keypt_kps,
        #    self.strides, self.scales)

    def forward_single(self, x, cls_convs, reg_convs, offset_convs,
            keypt_cls, keypt_hm, keypt_hm_kp, keypt_kps, keypt_offset, stride, scale):
        cls_feat = x
        reg_feat = x
        offset_feat = x

        for cls_layer in cls_convs:
            cls_feat = cls_layer(cls_feat)

        for reg_layer in reg_convs:
            reg_feat = reg_layer(reg_feat)

        for offset_layer in offset_convs:
            offset_feat = offset_layer(offset_feat)

        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        cls_score = keypt_cls(cls_feat).float()
        hm_score = keypt_hm(cls_feat).float()
        hm_kp_score = keypt_hm_kp(cls_feat).float()
        kps_pred = scale(keypt_kps(reg_feat)).float()
        kp_offset_pred = keypt_offset(offset_feat).sigmoid().float()

        return cls_score, hm_score, hm_kp_score, kps_pred, kp_offset_pred

    def debug(self, img_meta, hm, gt_hm, gt_hm_mask, hm_kp, gt_hm_kp, gt_hm_kp_mask, keypt, gt_keypt):
        #img = cv2.imread(img_meta['filename']).astype('float32')[480:, 80:1360, ...]
        img = cv2.imread(img_meta['filename']).astype('float32')
        if img_meta['flip']:
            img = cv2.flip(img, 1)
        crop_size = img_meta.get('img_crop', None)
        if crop_size is not None:
            img = img[crop_size[0]:crop_size[1], crop_size[2]:crop_size[3], :]
        imname = os.path.basename(img_meta['filename'])
        imname = os.path.splitext(imname)[0]
        debugger = Debugger(dataset='parkspace', ipynb=False, theme='white')
        #debugger.add_img(img, img_id='out_gt')
        # gt hm
        gt_hm = debugger.gen_colormap(gt_hm.cpu().detach().numpy())
        #gt_hm = debugger.gen_colormap((gt_hm_list[n][s] * gt_hm_mask_list[n][s]).cpu().detach().numpy())
        debugger.add_blend_img(img, gt_hm, 'parkspace_gt_hm')
        gt_hm_mask = debugger.gen_colormap(gt_hm_mask.cpu().detach().numpy())
        debugger.add_blend_img(img, gt_hm_mask, 'parkspace_gt_hm_mask')
        gt_hm_kp_mask = debugger.gen_colormap(gt_hm_kp_mask.cpu().detach().numpy())
        debugger.add_blend_img(img, gt_hm_kp_mask, 'parkspace_gt_hm_kp_mask')
        # pred hm
        pred_hm = debugger.gen_colormap(hm.detach().sigmoid().cpu().numpy())
        debugger.add_blend_img(img, pred_hm, 'parkspace_pred_hm')

        # hm_kp
        gt_hm_kp = debugger.gen_colormap_hp(gt_hm_kp.detach().cpu().numpy())
        pred_hm_kp = debugger.gen_colormap_hp(hm_kp.detach().sigmoid().cpu().numpy())
        debugger.add_blend_img(img, gt_hm_kp, 'parkspace_gt_hm_kp')
        debugger.add_blend_img(img, pred_hm_kp, 'parkspace_pred_hm_kp')

        dets = self.multi_pose_decode(
            heat=hm.detach().sigmoid().unsqueeze(0),
            bbox=None,
            kps=keypt.unsqueeze(0),
            hm_hp=hm_kp.detach().sigmoid().unsqueeze(0),
            hp_offset=None, K=50)

        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, 2:] *= self.strides[0]
        if isinstance(img_meta['scale_factor'], float):
            dets[:, :, 2:] /= img_meta['scale_factor']
        else:
            dets[..., 2::2] /= img_meta['scale_factor'][0]
            dets[..., 3::2] /= img_meta['scale_factor'][1]
        debugger.add_img(img, img_id='parkspace_pred_kps')
        for k in range(len(dets[0])):
            if dets[0, k, 0] > 0.3:
                debugger.add_coco_hp(dets[0, k, 6:], img_id='parkspace_pred_kps')
                #debugger.add_coco_bbox(dets[0, k, 2:6], dets[0, k, 1],
                #     dets[0, k, 0], img_id='pred_kps')
        # gt
        dets_gt = gt_keypt[..., 0:2].reshape(-1, self.num_keypts*2).unsqueeze(0).detach().cpu().numpy()
        if isinstance(img_meta['scale_factor'], float):
            dets_gt /= img_meta['scale_factor']
        else:
            dets_gt[..., ::2] /= img_meta['scale_factor'][0]
            dets_gt[..., 1::2] /= img_meta['scale_factor'][1]

        debugger.add_img(img, img_id='parkspace_gt_kps')
        for k in range(len(dets_gt[0])):
            debugger.add_coco_hp(dets_gt[0, k], img_id='parkspace_gt_kps')

        debugger.save_all_imgs('debug_info', prefix='{}_{}_'.format(self.iter, imname))

    @force_fp32(apply_to=('hm', 'hm_kp', 'kps'))
    def loss(self,
             cls,
             hm,
             hm_kp,
             kps,
             kp_offset,
             gt_keypts,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        assert len(cls) == len(hm) == len(hm_kp) == len(kps)
        featmap_sizes = [hm.size()[-2:]]

        if cfg['debug']:
            self.iter += 1

        gt_cls_list, gt_hm_list, gt_hm_mask_list, gt_hm_kp_list, gt_hm_kp_mask_list, gt_kps_list, gt_kps_mask_list, gt_kp_offset_list \
            = self.keypt_target(gt_keypts, gt_labels, featmap_sizes)

        loss_cls = torch.zeros(1, device=hm.device)
        loss_hm = torch.zeros(1, device=hm.device)
        loss_hm_kp = torch.zeros(1, device=hm.device)
        loss_kps = torch.zeros(1, device=hm.device)
        loss_offset = torch.zeros(1, device=hm.device)
        num_imgs = len(img_metas)
        for n in range(num_imgs):
            cls_pos_inds = gt_cls_list[n].reshape(-1).nonzero().squeeze(-1)
            hm_pos_inds = gt_hm_mask_list[n].reshape(-1).nonzero().squeeze(-1)
            hm_kp_pos_inds = gt_hm_kp_mask_list[n].reshape(-1).nonzero().squeeze(-1)
            kps_pos_inds = gt_kps_mask_list[n].reshape(-1).nonzero().squeeze(-1)
            offset_pos_inds = gt_kp_offset_list[n].reshape(-1).nonzero().squeeze(-1)

            if len(kps_pos_inds) > 0:
                kps_feat = (kps[n] * gt_kps_mask_list[n]).permute(1, 2, 0).contiguous().view(-1, self.num_keypts*2)
                kps_gt_feat = (gt_kps_list[n] * gt_kps_mask_list[n]).permute(1, 2, 0).contiguous().view(-1, self.num_keypts*2)
                loss_kps += self.loss_kps(kps_feat, kps_gt_feat) / gt_kps_mask_list[n].sum()

            if len(hm_kp_pos_inds) > 0:
                #flatten_gt_hm_kp = gt_hm_kp_list[n].reshape(-1)[hm_kp_pos_inds]
                #flatten_pred_hm_kp = hm_kp[n].reshape(-1)[hm_kp_pos_inds]
                flatten_gt_hm_kp = gt_hm_kp_list[n].reshape(-1)
                flatten_pred_hm_kp = hm_kp[n].reshape(-1)
                loss_hm_kp += self.loss_hm_kp(flatten_pred_hm_kp, flatten_gt_hm_kp)

            if len(hm_pos_inds) > 0: # parkspace heatmap noly use pos sample
                #flatten_gt_hm = gt_hm_list[n].reshape(-1)[hm_pos_inds]
                #flatten_pred_hm = hm[n].reshape(-1)[hm_pos_inds]
                flatten_gt_hm = gt_hm_list[n].reshape(-1)
                flatten_pred_hm = hm[n].reshape(-1)
                loss_hm += self.loss_hm(flatten_pred_hm, flatten_gt_hm)

            if len(cls_pos_inds) > 0:
                flatten_gt_cls = gt_cls_list[n].reshape(-1)[cls_pos_inds]
                flatten_pred_cls = cls[n].reshape(2, -1)[..., cls_pos_inds].permute(1, 0).contiguous()
                loss_cls += self.loss_cls(flatten_pred_cls, flatten_gt_cls)

            if len(offset_pos_inds) > 0:
                flatten_gt_offset = gt_kp_offset_list[n].reshape(-1)[offset_pos_inds]
                flatten_pred_offset = kp_offset[n].reshape(-1)[offset_pos_inds]
                loss_offset += self.loss_offset(flatten_pred_offset, flatten_gt_offset)

            #debug
            if cfg['debug'] and (self.iter % 1000 == 1):
                self.debug(img_metas[n], hm[n], gt_hm_list[n], gt_hm_mask_list[n], hm_kp[n], gt_hm_kp_list[n], gt_hm_kp_mask_list[n], kps[n], gt_keypts[n])

        return dict(
            loss_parkspace_cls=loss_cls,
            loss_parkspace_hm=loss_hm,
            loss_parkspace_hm_kp=loss_hm_kp,
            loss_parkspace_kps=loss_kps,
            loss_parkspace_offset=loss_offset)

    @force_fp32(apply_to=('hm', 'hm_kp', 'kps'))
    def get_parkspaces(self,
                   hm,
                   hm_kp,
                   kps,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(img_metas) == 1
        assert len(hm) == len(hm_kp) == len(kps)

        result_list = []
        extra_bbox_list = []
        featmap_sizes = [featmap.size()[-2:] for featmap in hm]
        
        '''
        dets = self.multi_pose_decode(
            heat=hm[0].detach().sigmoid().unsqueeze(0),
            bbox=None,
            kps=kps[0].unsqueeze(0),
            hm_hp=hm_kp[0].detach().sigmoid().unsqueeze(0),
            hp_offset=None, K=50)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, 2:] *= self.strides[0]
        dets[:, :, 2:] /= img_metas[0]['scale_factor']
        for k in range(len(dets[0])):
            if dets[0, k, 0] > 0.3:
                result_list.append(dets[0, k, 6:])
        '''

        heat=hm[0].detach().sigmoid().unsqueeze(0)
        kps=kps[0].unsqueeze(0)
        hm_hp=hm_kp[0].detach().sigmoid().unsqueeze(0)
        K=20

        batch, cat, height, width = heat.size()
        num_joints = kps.shape[1] // 2
        # heat = torch.sigmoid(heat)
        # perform nms on heatmaps
        heat = self._nms(heat, kernel=7)
        scores, inds, clses, ys, xs = self._topk(heat, K=K)

        kps = self._transpose_and_gather_feat(kps, inds)
        kps = kps.view(batch, K, num_joints * 2)
        kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
        kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
        xs = xs.view(batch, K, 1)
        ys = ys.view(batch, K, 1)
        clses  = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)

        #if hm_hp is not None:
        #    hm_hp = self._nms(hm_hp)
        #    thresh = 0.3
        #    kps = kps.view(batch, K, num_joints, 2).permute(
        #        0, 2, 1, 3).contiguous() # b x J x K x 2
        #    reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        #    hm_score, hm_inds, hm_ys, hm_xs = self._topk_channel(hm_hp, K=K) # b x J x K
        #    hm_xs = hm_xs + 0.5
        #    hm_ys = hm_ys + 0.5

        #    mask = (hm_score > thresh).float()
        #    hm_score = (1 - mask) * -1 + mask * hm_score
        #    hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        #    hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        #    hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
        #        2).expand(batch, num_joints, K, K, 2)
        #    dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        #    min_dist, min_ind = dist.min(dim=3) # b x J x K
        #    hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
        #    min_dist = min_dist.unsqueeze(-1)
        #    min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
        #        batch, num_joints, K, 1, 2)
        #    hm_kps = hm_kps.gather(3, min_ind)
        #    hm_kps = hm_kps.view(batch, num_joints, K, 2)
        #    mask = hm_score < thresh
        #    mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        #    #kps = (1 - mask) * hm_kps + mask * kps
        #    kps = kps.permute(0, 2, 1, 3).contiguous().view(
        #        batch, K, num_joints * 2)
        bboxes = torch.zeros([scores.shape[0], scores.shape[1], 4],
            dtype=scores.dtype, device=scores.device)
        detections = torch.cat([scores, clses, bboxes, kps], dim=2)

        dets = detections
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        for k in range(len(dets[0])):
            print('parkspace score: {}'.format(dets[0, k, 0]))
            if dets[0, k, 0] > 0.3:
                s = 0
                for p in range(self.num_keypts):
                    x = int(dets[0, k, 6+2*p])
                    y = int(dets[0, k, 6+2*p+1])
                    x = min(featmap_sizes[0][1] - 1, x)
                    y = min(featmap_sizes[0][0] - 1, y)
                    x = max(0, x)
                    y = max(0, y)
                    s += hm_hp[0, p, y, x]
                s /= self.num_keypts
                print('parkspace kp score: {}'.format(s))
                if s > 0.1:
                    dets[0, k, 2:] *= self.strides[0]
                    dets[0, k, 2:] /= img_metas[0]['scale_factor']
                    result_list.append(dets[0, k, 6:])

        '''
        for img_id in range(len(img_metas)):
            for s_id in range(num_levels):
                topK = min(500, featmap_sizes[s_id][0] * featmap_sizes[s_id][1])
                batch = kps[s_id].shape[0]
                hm[s_id] = self._nms(hm[s_id])
                scores, inds, clses, ys, xs = self._topk(hm[s_id], K=topK)
                clses  = clses.view(batch, topK, 1).float()
                scores = scores.view(batch, topK, 1)

                #debug
                #debug_hm = np.round(hm[s_id].squeeze().cpu().detach().numpy())
                #dst_name = 'debug_info/test_s{}_hm.jpg'.format(s_id)
                #cv2.imwrite(dst_name, debug_hm * 255 * 5)

                kps[s_id] = self._transpose_and_gather_feat(kps[s_id], inds)
                kps[s_id] = kps[s_id].view(batch, topK, self.num_keypts * 2)
                kp_xs = kps[s_id][..., 0:self.num_keypts]
                kp_ys = kps[s_id][..., self.num_keypts:]
                kp_xs += xs.view(batch, topK, 1).expand(batch, topK, self.num_keypts)
                kp_ys += ys.view(batch, topK, 1).expand(batch, topK, self.num_keypts)
                kps[s_id] = torch.stack([kp_xs, kp_ys], dim=-1).permute(0, 2, 1, 3).contiguous()
                reg_kps = kps[s_id].unsqueeze(3).expand(batch, self.num_keypts, topK, topK, 2)

                hm_kp[s_id] = self._nms(hm_kp[s_id])
                hm_scores, hm_inds, hm_ys, hm_xs = self._topk_channel(hm_kp[s_id], K=topK) # b x J x K

                kp_offset[s_id] = self._transpose_and_gather_feat(kp_offset[s_id], hm_inds.view(batch, -1))
                kp_offset[s_id] = kp_offset[s_id].view(batch, self.num_keypts, topK, 2)
                hm_xs = hm_xs + kp_offset[s_id][:, :, :, 0]
                hm_ys = hm_ys + kp_offset[s_id][:, :, :, 1]

                thresh = 0.1
                mask = (hm_scores > thresh).float()
                hm_scores = (1 - mask) * -1 + mask * hm_scores
                hm_ys = (1 - mask) * (-10000) + mask * hm_ys
                hm_xs = (1 - mask) * (-10000) + mask * hm_xs
                hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
                    2).expand(batch, self.num_keypts, topK, topK, 2)

                dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
                min_dist, min_ind = dist.min(dim=3) # b x J x K
                hm_scores = hm_scores.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
                min_dist = min_dist.unsqueeze(-1)
                min_ind = min_ind.view(batch, self.num_keypts, topK, 1, 1).expand(batch, self.num_keypts, topK, 1, 2)
                hm_kps = hm_kps.gather(3, min_ind)
                hm_kps = hm_kps.view(batch, self.num_keypts, topK, 2)

                mask = (hm_scores < thresh) + (min_dist > 50)

                mask = (mask > 0).float().expand(batch, self.num_keypts, topK, 2)
                reg_kps = (1 - mask) * hm_kps + mask * kps[s_id]
                reg_kps = reg_kps.permute(0, 2, 1, 3).contiguous().view(batch, topK, self.num_keypts * 2)

                reg_kps *= self.strides[s_id]
                result_list.append(reg_kps)
        '''

        return result_list, extra_bbox_list

    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _gather_feat(self, feat, ind, mask=None):
        dim  = feat.size(2)
        ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _topk(self, scores, K=40):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys   = (topk_inds // width).int().float()
        topk_xs   = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind // K).int()
        topk_inds = self._gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
    
    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _topk_channel(self, scores, K=40):
        batch, cat, height, width = scores.size()
      
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys   = (topk_inds // width).int().float()
        topk_xs   = (topk_inds % width).int().float()

        return topk_scores, topk_inds, topk_ys, topk_xs

    def multi_pose_decode(self, heat, bbox=None, kps=None, hm_hp=None, hp_offset=None, K=100):
        batch, cat, height, width = heat.size()
        num_joints = kps.shape[1] // 2
        # heat = torch.sigmoid(heat)
        # perform nms on heatmaps
        heat = self._nms(heat)
        scores, inds, clses, ys, xs = self._topk(heat, K=K)

        kps = self._transpose_and_gather_feat(kps, inds)
        kps = kps.view(batch, K, num_joints * 2)
        kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
        kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
        xs = xs.view(batch, K, 1)
        ys = ys.view(batch, K, 1)
        if bbox is not None:
            bbox = self._transpose_and_gather_feat(bbox, inds)
            bbox = bbox.view(batch, K, 4)
            bboxes = torch.cat([xs - bbox[..., 0].unsqueeze(2), 
                                ys - bbox[..., 1].unsqueeze(2),
                                xs + bbox[..., 2].unsqueeze(2), 
                                ys + bbox[..., 3].unsqueeze(2)], dim=2)

        clses  = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)

        if hm_hp is not None:
            hm_hp = self._nms(hm_hp)
            thresh = 0.3
            kps = kps.view(batch, K, num_joints, 2).permute(
                0, 2, 1, 3).contiguous() # b x J x K x 2
            reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
            hm_score, hm_inds, hm_ys, hm_xs = self._topk_channel(hm_hp, K=K) # b x J x K
            if hp_offset is not None:
                hp_offset = self._transpose_and_gather_feat(
                    hp_offset, hm_inds.view(batch, -1))
                hp_offset = hp_offset.view(batch, num_joints, K, 2)
                hm_xs = hm_xs + hp_offset[:, :, :, 0]
                hm_ys = hm_ys + hp_offset[:, :, :, 1]
            else:
                hm_xs = hm_xs + 0.5
                hm_ys = hm_ys + 0.5

            mask = (hm_score > thresh).float()
            hm_score = (1 - mask) * -1 + mask * hm_score
            hm_ys = (1 - mask) * (-10000) + mask * hm_ys
            hm_xs = (1 - mask) * (-10000) + mask * hm_xs
            hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
                2).expand(batch, num_joints, K, K, 2)
            dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
            min_dist, min_ind = dist.min(dim=3) # b x J x K
            hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
            min_dist = min_dist.unsqueeze(-1)
            min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
                batch, num_joints, K, 1, 2)
            hm_kps = hm_kps.gather(3, min_ind)
            hm_kps = hm_kps.view(batch, num_joints, K, 2)
            if bbox is not None:
                l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
                t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
                r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
                b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
                mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
                       (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
                       (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
            else:
                mask = hm_score < thresh
            mask = (mask > 0).float().expand(batch, num_joints, K, 2)
            #kps = (1 - mask) * hm_kps + mask * kps
            kps = kps.permute(0, 2, 1, 3).contiguous().view(
                batch, K, num_joints * 2)
        if bbox is not None:
            detections = torch.cat([scores, clses, bboxes, kps], dim=2)
        else:
            bboxes = torch.zeros([scores.shape[0], scores.shape[1], 4],
                dtype=scores.dtype, device=scores.device)
            detections = torch.cat([scores, clses, bboxes, kps], dim=2)

        return detections

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          extra_cls_scores,
                          extra_face_preds,
                          extra_centernesses,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_extra_faces = []
        mlvl_extra_scores = []
        mlvl_extra_centerness = []
        for cls_score, bbox_pred, centerness, extra_cls_score, extra_face_pred, extra_centerness, ori_points in zip(
                cls_scores, bbox_preds, centernesses, extra_cls_scores, extra_face_preds, extra_centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert extra_cls_score.size()[-2:] == extra_face_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            extra_scores = extra_cls_score.permute(1, 2, 0).reshape(-1, self.extra_cls_out_channels).sigmoid()
            extra_centerness = extra_centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            extra_face_pred = extra_face_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = ori_points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]

                extra_max_scores, _ = (extra_scores * extra_centerness[:, None]).max(dim=1)
                _, extra_topk_inds = extra_max_scores.topk(nms_pre)
                extra_points = ori_points[extra_topk_inds, :]
                extra_face_pred = extra_face_pred[extra_topk_inds, :]
                extra_scores = extra_scores[extra_topk_inds, :]
                extra_centerness = extra_centerness[extra_topk_inds]
            else:
                points = ori_points
                extra_points = ori_points
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            extra_faces = distance2bbox(extra_points, extra_face_pred, max_shape=img_shape)
            #extra_keypts = distance2keypt(points, extra_keypt_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_extra_faces.append(extra_faces)
            mlvl_extra_scores.append(extra_scores)
            mlvl_extra_centerness.append(extra_centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_extra_faces = torch.cat(mlvl_extra_faces)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_extra_faces /= mlvl_extra_faces.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        mlvl_extra_scores = torch.cat(mlvl_extra_scores)
        extra_padding = mlvl_extra_scores.new_zeros(mlvl_extra_scores.shape[0], 1)
        mlvl_extra_scores = torch.cat([extra_padding, mlvl_extra_scores], dim=1)
        mlvl_extra_centerness = torch.cat(mlvl_extra_centerness)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        det_extra_faces, det_extra_labels = multiclass_nms(
            mlvl_extra_faces,
            mlvl_extra_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_extra_centerness)
        #det_bboxes, det_labels, det_keypts = nms_with_extra(
        #    mlvl_bboxes,
        #    mlvl_scores,
        #    mlvl_extra_keypts,
        #    mlvl_extra_scores,
        #    cfg.score_thr,
        #    cfg.nms,
        #    cfg.max_per_img,
        #    score_factors=mlvl_centerness)
        return det_bboxes, det_labels, det_extra_faces, det_extra_labels

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def keypt_target(self, keypts, labels, featmap_sizes):
        cls_list, hm_list, hm_mask_list, hm_kp_list, hm_kp_mask_list, kps_list, kps_mask_list, kp_offset_list \
            = multi_apply(self.keypt_target_single, keypts, labels,
                strides=self.strides, featmap_sizes=featmap_sizes)
        return cls_list, hm_list, hm_mask_list, hm_kp_list, hm_kp_mask_list, kps_list, kps_mask_list, kp_offset_list
    
    def keypt_target_single(self, keypts, labels, strides, featmap_sizes):
        assert len(self.strides) == len(featmap_sizes) == 1
        num_points = featmap_sizes[0][0] * featmap_sizes[0][1]
        num_gts = keypts.size(0)
        if num_gts == 0:
            return torch.zeros(self.heads['hm'], featmap_sizes[0][0], featmap_sizes[0][1]), \
                   torch.zeros(1, featmap_sizes[0][0], featmap_sizes[0][1]), \
                   torch.zeros(self.heads['hm_kp'], featmap_sizes[0][0], featmap_sizes[0][1]), \
                   torch.zeros(self.heads['hm_kp'], featmap_sizes[0][0], featmap_sizes[0][1]), \
                   torch.zeros(self.heads['kps'], featmap_sizes[0][0], featmap_sizes[0][1]), \
                   torch.zeros(self.heads['kps'], featmap_sizes[0][0], featmap_sizes[0][1]) 

        kpts = keypts.clone()
        kpts[..., 0:2] = keypts[..., 0:2] / self.strides[0] #uniform to feature map size
        kptx_int = torch.floor(kpts[..., 0])
        kpty_int = torch.floor(kpts[..., 1])
        kp_offset_x = kpts[..., 0] - kptx_int
        kp_offset_y = kpts[..., 1] - kpty_int

        #y_min = torch.min(kpts[:, 0, 1], kpts[:, 2, 1])
        #y_max = torch.max(kpts[:, 0, 1], kpts[:, 2, 1])
        x_min = kpts[:, :, 0].min(dim=1)[0]
        x_max = kpts[:, :, 0].max(dim=1)[0]
        y_min = kpts[:, :, 1].min(dim=1)[0]
        y_max = kpts[:, :, 1].max(dim=1)[0]
        w = x_max - x_min
        h = y_max - y_min
        ctx = kpts[:, :, 0].mean(dim=1)
        cty = kpts[:, :, 1].mean(dim=1)
        ctx_int = torch.floor(ctx).type(torch.int32)
        cty_int = torch.floor(cty).type(torch.int32)
        regx = ctx - ctx_int
        regy = cty - cty_int
        areas = w * h
        areas = areas[None].repeat(num_points, 1)

        x_range = torch.range(0, featmap_sizes[0][1] - 1, device=kpts.device)
        y_range = torch.range(0, featmap_sizes[0][0] - 1, device=kpts.device)
        ys, xs = torch.meshgrid(y_range, x_range)
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - x_min
        right = x_max - xs
        top = ys - y_min
        bottom = y_max - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        areas[inside_gt_bbox_mask == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        bbox_targets = bbox_targets.reshape(featmap_sizes[0][0], featmap_sizes[0][1], self.heads['bbox']).permute(2, 0, 1)

        '''
        left_right = torch.stack((left, right), -1)
        left_right = left_right[range(num_points), min_area_inds]
        top_bottom = torch.stack((top, bottom), -1)
        top_bottom = top_bottom[range(num_points), min_area_inds]
        hm_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        hm_targets[min_area == INF] = 0
        hm_targets = hm_targets.reshape(featmap_sizes[0][0], featmap_sizes[0][1], self.heads['hm']).permute(2, 0, 1)
        '''
        hm_targets = torch.zeros(1, featmap_sizes[0][0], featmap_sizes[0][1], dtype=kpts.dtype, device=kpts.device)
        cls_targets = torch.zeros(1, featmap_sizes[0][0], featmap_sizes[0][1], dtype=torch.int64, device=kpts.device)

        hm_mask_targets = torch.ones(xs.size(0), dtype=torch.bool, device=kpts.device)
        hm_mask_targets[min_area == INF] = False
        hm_mask_targets = hm_mask_targets.reshape(1, featmap_sizes[0][0], featmap_sizes[0][1])

        # for keypts
        x0 = kpts[..., 0, 0] - xs
        y0 = kpts[..., 0, 1] - ys
        x1 = kpts[..., 1, 0] - xs
        y1 = kpts[..., 1, 1] - ys
        x2 = kpts[..., 2, 0] - xs
        y2 = kpts[..., 2, 1] - ys
        x3 = kpts[..., 3, 0] - xs
        y3 = kpts[..., 3, 1] - ys
        kpt_stacks = torch.stack((x0, y0, x1, y1, x2, y2, x3, y3), -1)
        kpt_stacks = kpt_stacks.reshape(featmap_sizes[0][0], featmap_sizes[0][1], kpts.shape[0], self.heads['kps']).permute(3, 0, 1, 2)
        #kpt_targets = torch.stack((x0, y0, x1, y1, x2, y2, x3, y3), -1)
        #kpt_targets = kpt_targets[range(num_points), min_area_inds]
        #kpt_targets = kpt_targets.reshape(featmap_sizes[0][0], featmap_sizes[0][1], self.heads['kps']).permute(2, 0, 1)
        kpt_targets = torch.zeros(
            (self.heads['kps'], featmap_sizes[0][0], featmap_sizes[0][1]),
            dtype=torch.float32, device=kpts.device)
        offset_targets = torch.zeros(
            (self.heads['kp_offset'], featmap_sizes[0][0], featmap_sizes[0][1]),
            dtype=torch.float32, device=kpts.device)

        hp_radius = self.gaussian_radius_cuda((torch.ceil(h), torch.ceil(w)))# * 0.5 # 乘以0.5系数，减小绘制半径，使得回归的区域更准一些

        assert self.heads['hm_kp'] == self.num_keypts
        hm_kp_targets = torch.zeros(
            (self.heads['hm_kp'], featmap_sizes[0][0], featmap_sizes[0][1]),
            dtype=torch.float32, device=kpts.device)
        hm_kp_mask_targets = torch.zeros(
            (self.heads['hm_kp'], featmap_sizes[0][0], featmap_sizes[0][1]),
            dtype=torch.bool, device=kpts.device)
        kps_mask_targets = torch.zeros(
            (self.heads['kps'], featmap_sizes[0][0], featmap_sizes[0][1]),
            dtype=torch.bool, device=kpts.device)

        for j in range(kpts.shape[0]): # object num
            if min(w[j], h[j]) < 2:
                continue
            xl = int(max(0, ctx_int[j] - hp_radius[j]))
            xr = int(min(featmap_sizes[0][1], ctx_int[j] + hp_radius[j]))
            yt = int(max(0, cty_int[j] - hp_radius[j]))
            yb = int(min(featmap_sizes[0][0], cty_int[j] + hp_radius[j]))
            kpt_targets[:, yt:yb+1, xl:xr+1] =  kpt_stacks[:, yt:yb+1, xl:xr+1, j]
            self.draw_umich_gaussian(hm_targets[0], (int(ctx_int[j].detach().cpu()), int(cty_int[j].detach().cpu())), int(hp_radius[j]))
            cls_targets[0, yt:yb+1, xl:xr+1] = labels[j]
            for k in range(self.num_keypts):
                if hp_radius[j] > 0:
                    if int(kpts[j, k, 2]) == 3: # x and y realiable
                        self.draw_umich_gaussian(hm_kp_targets[k],
                            (int(kptx_int[j][k].detach().cpu()), int(kpty_int[j][k].detach().cpu())),
                            int(hp_radius[j].detach().cpu()))
                        hm_kp_mask_targets[k, int(y_min[j].floor()):int(y_max[j].ceil())+1, int(x_min[j].floor()):int(x_max[j].ceil())+1] = True
                        #kps_mask_targets[2*k + 0, cty_int[j], ctx_int[j]] = True
                        #kps_mask_targets[2*k + 1, cty_int[j], ctx_int[j]] = True
                        kps_mask_targets[2*k + 0, yt:yb+1, xl:xr+1] = True
                        kps_mask_targets[2*k + 1, yt:yb+1, xl:xr+1] = True
                    elif int(kpts[j, k, 2]) == 2: #noly y realiable
                        #kps_mask_targets[2*k + 1, cty_int[j], ctx_int[j]] = True
                        kps_mask_targets[2*k + 1, yt:yb+1, xl:xr+1] = True
                    elif int(kpts[j, k, 2]) == 1: #noly x realiable
                        #kps_mask_targets[2*k + 0, cty_int[j], ctx_int[j]] = True
                        kps_mask_targets[2*k + 0, yt:yb+1, xl:xr+1] = True
                offset_targets[0, int(kpty_int[j, k]), int(kptx_int[j, k])] = kp_offset_x[j, k]
                offset_targets[1, int(kpty_int[j, k]), int(kptx_int[j, k])] = kp_offset_y[j, k]

        return cls_targets, hm_targets, hm_mask_targets, hm_kp_targets, hm_kp_mask_targets, kpt_targets, kps_mask_targets, offset_targets

    '''
    def keypt_target_single(self, keypts, points, strides, featmap_sizes):
        assert len(self.strides) == 1
        hm_list = []
        hm_mask_list = []
        hm_kp_list = []
        hm_kp_mask_list = []
        kps_list = []
        kps_mask_list = []
        for i, stride in enumerate(strides):
            hm = torch.zeros(
                (self.heads['hm'], featmap_sizes[i][0], featmap_sizes[i][1]),
                dtype=torch.float32, device=keypts.device)
            hm_mask = torch.zeros(
                (1, featmap_sizes[i][0], featmap_sizes[i][1]), 
                dtype=torch.bool, device=keypts.device)
            hm_kp = torch.zeros(
                (self.heads['hm_kp'], featmap_sizes[i][0], featmap_sizes[i][1]),
                dtype=torch.float32, device=keypts.device)
            hm_kp_mask = torch.zeros(
                (self.num_keypts, featmap_sizes[i][0], featmap_sizes[i][1]),
                dtype=torch.bool, device=keypts.device)
            #kps = torch.zeros([self.num_keypts * 2, featmap_sizes[i][0], featmap_sizes[i][1]], dtype=torch.float32, device=keypts.device)
            kps_mask = torch.zeros(
                (self.heads['kps'], featmap_sizes[i][0], featmap_sizes[i][1]),
                dtype=torch.bool, device=keypts.device)
            #reg = torch.zeros([4, featmap_sizes[i][0], featmap_sizes[i][1]], dtype=torch.float32, device=keypts.device)

            kpts = keypts.clone()
            kpts[..., 0:2] = keypts[..., 0:2] / stride #uniform to feature map size
            #kpsx = kpts[:, :, 0] - ctx_int[:, None].expand(kpts.shape[0], 4)
            #kpsy = kpts[:, :, 1] - cty_int[:, None].expand(kpts.shape[0], 4)
            kptx_int = torch.floor(kpts[..., 0])
            kpty_int = torch.floor(kpts[..., 1])
            kp_offset_x = kpts[..., 0] - kptx_int
            kp_offset_y = kpts[..., 1] - kpty_int

            # for bbox
            num_gts = kpts.shape[0]
            bboxes = torch.zeros([num_gts, 4], dtype=torch.float32, device=kpts.device)
            bboxes[..., 1] = torch.min(kpts[:, 0, 1], kpts[:, 2, 1])
            bboxes[..., 3] = torch.max(kpts[:, 0, 1], kpts[:, 2, 1])
            for t in range(num_gts):
                bboxes[t, 0] = kpts[t, :, 0][kpts[t, :, 2] > 0].min()
                bboxes[t, 2] = kpts[t, :, 0][kpts[t, :, 2] > 0].max()
            w = bboxes[..., 2] - bboxes[..., 0]
            h = bboxes[..., 3] - bboxes[..., 1]
            ctx = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
            cty = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
            ctx_int = torch.floor(ctx).type(torch.int32)
            cty_int = torch.floor(cty).type(torch.int32)
            regx = ctx - ctx_int
            regy = cty - cty_int

            areas = w * h
            num_points = featmap_sizes[i][0] * featmap_sizes[i][1]
            areas = areas[None].repeat(num_points, 1)
            x_range = torch.range(0, featmap_sizes[i][1] - 1, device=kpts.device)
            y_range = torch.range(0, featmap_sizes[i][0] - 1, device=kpts.device)
            ys, xs = torch.meshgrid(y_range, x_range)
            xs = xs.reshape(-1)
            ys = ys.reshape(-1)
            xs = xs[:, None].expand(num_points, num_gts)
            ys = ys[:, None].expand(num_points, num_gts)
            left = xs - bboxes[..., 0]
            right = bboxes[..., 2] - xs
            top = ys - bboxes[..., 1]
            bottom = bboxes[..., 3] - ys
            bbox_targets = torch.stack((left, top, right, bottom), -1)
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
            areas[inside_gt_bbox_mask == 0] = INF
            min_area, min_area_inds = areas.min(dim=1)
            bbox_targets = bbox_targets[range(num_points), min_area_inds]
            bbox_targets = bbox_targets.reshape(featmap_sizes[i][0], featmap_sizes[i][1], 4).permute(2, 0, 1)

            # for keypts
            x0 = kpts[..., 0, 0] - xs
            y0 = kpts[..., 0, 1] - ys
            x1 = kpts[..., 1, 0] - xs
            y1 = kpts[..., 1, 1] - ys
            x2 = kpts[..., 2, 0] - xs
            y2 = kpts[..., 2, 1] - ys
            x3 = kpts[..., 3, 0] - xs
            y3 = kpts[..., 3, 1] - ys
            x4 = kpts[..., 4, 0] - xs
            y4 = kpts[..., 4, 1] - ys
            x5 = kpts[..., 5, 0] - xs
            y5 = kpts[..., 5, 1] - ys
            kpt_targets = torch.stack((x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5), -1)
            kpt_targets = kpt_targets[range(num_points), min_area_inds]
            kpt_targets = kpt_targets.reshape(featmap_sizes[i][0], featmap_sizes[i][1], self.num_keypts*2).permute(2, 0, 1)

            for j in range(kpts.shape[0]): # object num
                if min(w[j], h[j]) < 2:
                    continue
                hp_radius = int(self.gaussian_radius((math.ceil(h[j]), math.ceil(w[j]))))
                if hp_radius > 0:
                    self.draw_umich_gaussian(hm[0], (ctx_int[j], cty_int[j]), hp_radius)
                    #self.draw_umich_gaussian(hm_mask[0], (ctx_int[j], cty_int[j]), hp_radius*2)
                    x_s = max(0, int(ctx[j] - w[j] * 0.5))
                    x_e = min(featmap_sizes[i][1], int(ctx[j] + w[j] * 0.5 + 0.5))
                    y_s = max(0, int(cty[j] - h[j] * 0.5))
                    y_e = min(featmap_sizes[i][0], int(cty[j] + h[j] * 0.5 + 0.5))
                    hm_mask[0, y_s:y_e, x_s:x_e] = True
                #hm[0, cty_int[j], ctx_int[j]] = 0.9999 
                #hm_mask[0, cty_int[j], ctx_int[j]] = 1.0

                for k in range(self.num_keypts):
                    if hp_radius > 0:
                        if int(kpts[j, k, 2]) == 3: # x and y realiable
                            self.draw_umich_gaussian(hm_kp[k], (int(kptx_int[j][k]), int(kpty_int[j][k])), hp_radius)
                            #self.draw_umich_gaussian(hm_kp_mask[k], (int(kptx_int[j][k]), int(kpty_int[j][k])), int(hp_radius*2))
                            hm_kp_mask[k, y_s:y_e, x_s:x_e] = True
                            kps_mask[2*k + 0, cty_int[j], ctx_int[j]] = True
                            kps_mask[2*k + 1, cty_int[j], ctx_int[j]] = True
                        elif int(kpts[j, k, 2]) == 2: #noly y realiable
                            kps_mask[2*k + 1, cty_int[j], ctx_int[j]] = True
                        elif int(kpts[j, k, 2]) == 1: #noly x realiable
                            kps_mask[2*k + 0, cty_int[j], ctx_int[j]] = True

            hm_list.append(hm)
            hm_mask_list.append(hm_mask)
            hm_kp_list.append(hm_kp)
            hm_kp_mask_list.append(hm_kp_mask)
            kps_list.append(kpt_targets)
            kps_mask_list.append(kps_mask)
        return hm_list, hm_mask_list, hm_kp_list, hm_kp_mask_list, kps_list, kps_mask_list
    '''

    def gaussian_radius_cuda(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2

        r = torch.stack((r1, r2, r3), -1)
        return r.min(-1)[0]

    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1,-n:n+1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)
  
        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
            out_masked_heatmap = masked_heatmap.cpu().numpy().copy()
            np.maximum(masked_heatmap.cpu().numpy(), masked_gaussian * k, out=out_masked_heatmap)
            heatmap[y - top:y + bottom, x - left:x + right] = torch.tensor(out_masked_heatmap, dtype=heatmap.dtype, device=heatmap.device)
        #return heatmap
