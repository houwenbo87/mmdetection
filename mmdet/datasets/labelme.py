import os.path as osp
import xml.etree.ElementTree as ET
import json

import re
import cv2
import mmcv
import math
import numpy as np

from progress.bar import Bar
from easydict import EasyDict as edict

from .custom import CustomDataset
from .builder import DATASETS

@DATASETS.register_module
class LabelmeDataset(CustomDataset):
    CLASSES = ('idle parking spot', 'occupied parking spot')

    def __init__(self, min_size=None, **kwargs):
        super(LabelmeDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size

    def load_annotations(self, ann_file):
        self.img_infos = []
        name_list = mmcv.list_from_file(ann_file)
        #bar = Bar('load labels', max=len(xml_files), suffix='%(percent)d%%')
        bar = Bar('load labels', max=len(name_list))
        for id in name_list:
            bar.next()
            impath = osp.join(self.img_prefix, id + '.jpg')
            annpath = osp.join(self.img_prefix, id + '.json')

            # im properties
            im = mmcv.imread(impath, 'color')
            width = im.shape[1]
            height = im.shape[0]

            self.img_infos.append(
                dict(id=id, filename=impath, width=width, height=height))
        bar.finish()
        print('total label num: {}'.format(len(self.img_infos)))
        return self.img_infos

    def get_ann_info(self, idx):
        id = self.img_infos[idx]['id']
        impath = osp.join(self.img_prefix, id + '.jpg')
        annpath = osp.join(self.img_prefix, id + '.json')
        bboxes = []
        labels = []
        parkspaces = []
        parkspace_labels = []
        with open(annpath) as f:
            data = json.load(f)
        for p in data['shapes']:
            label = p['label']
            pts = p['points']
            assert len(pts) >= 4
            pkpt_list = []
            pkpt_list.append([pts[0][0], pts[0][1], 3])
            pkpt_list.append([pts[1][0], pts[1][1], 3])
            pkpt_list.append([pts[-2][0], pts[-2][1], 3])
            pkpt_list.append([pts[-1][0], pts[-1][1], 3])
            parkspaces.append(pkpt_list)
            parkspace_labels.append(self.cat2label[label])

            xmax = max(pts[0][0], pts[1][0], pts[-2][0], pts[-1][0])
            ymax = max(pts[0][1], pts[1][1], pts[-2][1], pts[-1][1])
            xmin = min(pts[0][0], pts[1][0], pts[-2][0], pts[-1][0])
            ymin = min(pts[0][1], pts[1][1], pts[-2][1], pts[-1][1])

            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.cat2label[label])

        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)

        if not parkspaces:
            parkspaces = np.zeros((0, 4, 3))
            parkspace_labels = np.zeros((0, ))
        else:
            parkspaces = np.array(parkspaces, ndmin=3)
            parkspace_labels = np.array(parkspace_labels)
        
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            parkspaces=parkspaces.astype(np.float32),
            parkspace_labels=parkspace_labels.astype(np.int64))
        return ann

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        """Evaluation in COCO protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str: float]
        """

        return {'ap':0}
