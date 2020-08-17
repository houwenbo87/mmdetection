import os.path as osp
import xml.etree.ElementTree as ET

import cv2
import mmcv
import numpy as np

from progress.bar import Bar

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class Object2DDataset(XMLDataset):
    CLASSES = ('vehicle', 'tricycle', 'pedestrian', 'ground_mark', 'traffic_sign', 'bicycle', 'motorcycle', 'cone')

    def __init__(self, min_size=None, **kwargs):
        super(Object2DDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size

    def load_annotations(self, ann_file):
        self.label_path = {}
        self.img_infos = []
        xml_files = mmcv.list_from_file(ann_file)
        #bar = Bar('load labels', max=len(xml_files), suffix='%(percent)d%%')
        bar = Bar('load labels', max=len(xml_files))
        for id, xml_file in enumerate(xml_files):
            bar.next()
            xml_path = osp.join(self.img_prefix, xml_file)
            try:
                tree = ET.parse(xml_path)
            except:
                print('Can not open {}\n'.format(xml_path))
                pass
            else:
                root = tree.getroot()
                impath = root.find('path').text
                fullpath = osp.join(self.img_prefix, impath)
                exists = osp.isfile(fullpath)
                if not exists:
                    continue
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                #id = root.find('filename').text
                self.img_infos.append(
                    dict(id=id, filename=impath, width=width, height=height))
                self.label_path[id] = xml_path
        bar.finish()
        print('total label num: {}'.format(len(self.label_path)))
        return self.img_infos

    def get_ann_info(self, idx):
        id = self.img_infos[idx]['id']
        xml_path = self.label_path[id]
        gt_bboxes = []
        gt_labels = []
        try:
            tree = ET.parse(xml_path)
        except:
            print('Can not open {}\n'.format(xml_path))
            pass
        else:
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            objects = root.find('objects')
            vehicles = objects.findall('vehicle')
            for ins_id, obj in enumerate(vehicles):
                name = obj.find('name').text
                label = self.cat2label['vehicle']

                bnd_box = obj.find('bndbox')
                bbox = [
                    min(float(bnd_box.find('xmin').text), float(bnd_box.find('xmax').text)),
                    min(float(bnd_box.find('ymin').text), float(bnd_box.find('ymax').text)),
                    max(float(bnd_box.find('xmin').text), float(bnd_box.find('xmax').text)),
                    max(float(bnd_box.find('ymin').text), float(bnd_box.find('ymax').text))
                ]

                gt_bboxes.append(bbox)
                gt_labels.append(label)

            tricycles = objects.findall('tricycle')
            for ins_id, obj in enumerate(tricycles):
                name = obj.find('name').text
                label = self.cat2label['tricycle']

                bnd_box = obj.find('bndbox')
                bbox = [
                    min(float(bnd_box.find('xmin').text), float(bnd_box.find('xmax').text)),
                    min(float(bnd_box.find('ymin').text), float(bnd_box.find('ymax').text)),
                    max(float(bnd_box.find('xmin').text), float(bnd_box.find('xmax').text)),
                    max(float(bnd_box.find('ymin').text), float(bnd_box.find('ymax').text))
                ]

                gt_bboxes.append(bbox)
                gt_labels.append(label)


            pedestrians = objects.findall('pedestrian')
            for ins_id, obj in enumerate(pedestrians):
                name = obj.find('name').text
                label = self.cat2label['pedestrian']

                bnd_box = obj.find('bndbox')
                bbox = [
                    min(float(bnd_box.find('xmin').text), float(bnd_box.find('xmax').text)),
                    min(float(bnd_box.find('ymin').text), float(bnd_box.find('ymax').text)),
                    max(float(bnd_box.find('xmin').text), float(bnd_box.find('xmax').text)),
                    max(float(bnd_box.find('ymin').text), float(bnd_box.find('ymax').text))
                ]

                gt_bboxes.append(bbox)
                gt_labels.append(label)

            ground_marks = objects.findall('ground_mark')
            for ins_id, obj in enumerate(ground_marks):
                name = obj.find('name').text
                label = self.cat2label['ground_mark']

                bnd_box = obj.find('bndbox')
                bbox = [
                    min(float(bnd_box.find('xmin').text), float(bnd_box.find('xmax').text)),
                    min(float(bnd_box.find('ymin').text), float(bnd_box.find('ymax').text)),
                    max(float(bnd_box.find('xmin').text), float(bnd_box.find('xmax').text)),
                    max(float(bnd_box.find('ymin').text), float(bnd_box.find('ymax').text))
                ]

                gt_bboxes.append(bbox)
                gt_labels.append(label)

            traffic_signs = objects.findall('traffic_sign')
            for ins_id, obj in enumerate(traffic_signs):
                name = obj.find('name').text
                label = self.cat2label['traffic_sign']

                bnd_box = obj.find('bndbox')
                bbox = [
                    min(float(bnd_box.find('xmin').text), float(bnd_box.find('xmax').text)),
                    min(float(bnd_box.find('ymin').text), float(bnd_box.find('ymax').text)),
                    max(float(bnd_box.find('xmin').text), float(bnd_box.find('xmax').text)),
                    max(float(bnd_box.find('ymin').text), float(bnd_box.find('ymax').text))
                ]

                gt_bboxes.append(bbox)
                gt_labels.append(label)

            bicycles = objects.findall('bicycle')
            for ins_id, obj in enumerate(bicycles):
                name = obj.find('name').text
                label = self.cat2label['bicycle']

                bnd_box = obj.find('bndbox')
                bbox = [
                    min(float(bnd_box.find('xmin').text), float(bnd_box.find('xmax').text)),
                    min(float(bnd_box.find('ymin').text), float(bnd_box.find('ymax').text)),
                    max(float(bnd_box.find('xmin').text), float(bnd_box.find('xmax').text)),
                    max(float(bnd_box.find('ymin').text), float(bnd_box.find('ymax').text))
                ]

                gt_bboxes.append(bbox)
                gt_labels.append(label)

            motorcycles = objects.findall('motorcycle')
            for ins_id, obj in enumerate(motorcycles):
                name = obj.find('name').text
                label = self.cat2label['motorcycle']

                bnd_box = obj.find('bndbox')
                bbox = [
                    min(float(bnd_box.find('xmin').text), float(bnd_box.find('xmax').text)),
                    min(float(bnd_box.find('ymin').text), float(bnd_box.find('ymax').text)),
                    max(float(bnd_box.find('xmin').text), float(bnd_box.find('xmax').text)),
                    max(float(bnd_box.find('ymin').text), float(bnd_box.find('ymax').text))
                ]

                gt_bboxes.append(bbox)
                gt_labels.append(label)

            cones = objects.findall('cone')
            for ins_id, obj in enumerate(cones):
                name = obj.find('name').text
                label = self.cat2label['cone']

                bnd_box = obj.find('bndbox')
                bbox = [
                    min(float(bnd_box.find('xmin').text), float(bnd_box.find('xmax').text)),
                    min(float(bnd_box.find('ymin').text), float(bnd_box.find('ymax').text)),
                    max(float(bnd_box.find('xmin').text), float(bnd_box.find('xmax').text)),
                    max(float(bnd_box.find('ymin').text), float(bnd_box.find('ymax').text))
                ]

                gt_bboxes.append(bbox)
                gt_labels.append(label)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            )

        return ann
