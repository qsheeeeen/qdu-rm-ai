import logging
import os

import numpy as np
from gluoncv import data as gdata

try:
    import xml.etree.cElementTree as ETree
except ImportError:
    import xml.etree.ElementTree as ETree


class DJIROCODetection(gdata.VOCDetection):
    """Base class for custom Dataset which follows protocol/formatting of the well-known VOC object detection dataset.

    Parameters
    ----------
    root : str, default '~/Dataset/DJI ROCO'
        Path to folder storing the dataset.
    target : str, default 'all'
        How dataset are generated.
    splits : list of tuples, default
        (('central', 'trainval'), ('north', 'trainval'), ('south', 'trainval'), ('final', 'trainval'))
        List of combinations of (competition, name)
        For competition, candidates can be: 'central', 'final', 'north', 'south'.
        For names, candidates can be: 'trainval', 'test'.
    transform : callable, default = None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.
        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, default = None
        By default, the N classes are mapped into indices from 0 to N-1. We can
        customize it by providing a str to int dict specifying how to map class
        names to indices. This is only for advanced users, when you want to swap the orders
        of class labels.
    preload_label : bool, default = True
        If True, then parse and load all labels into memory during
        initialization. It often accelerate speed but require more memory
        usage. Typical preloaded labels took tens of MB. You only need to disable it
        when your dataset is extremely large.
    """

    def __init__(self, root=os.path.join('~', 'Dataset', 'DJI ROCO'), target='all',
                 splits=(('central', 'trainval'), ('north', 'trainval'), ('south', 'trainval'), ('final', 'trainval')),
                 transform=None, index_map=None, preload_label=True):

        if target == 'all':
            self.CLASSES = ('car', 'watcher', 'base', 'ignore', 'armor')
        elif target == 'armor':
            self.CLASSES = ('armor',)
        elif target == 'robot':
            self.CLASSES = ('car', 'watcher', 'base')
        else:
            raise ValueError('target:{} Not implemented.'.format(target))

        super(DJIROCODetection, self).__init__(root=os.path.expanduser(root),
                                               splits=splits,
                                               transform=transform,
                                               index_map=index_map,
                                               preload_label=preload_label)

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        root = ETree.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            try:
                difficult = int(obj.findtext('difficulty'))
            except ValueError:
                difficult = 0
            except TypeError:
                difficult = 0
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = min(max((float(xml_box.find('xmin').text)), 0), width - 1)
            ymin = min(max((float(xml_box.find('ymin').text)), 0), height - 1)
            xmax = min(max((float(xml_box.find('xmax').text)), xmin), width)
            ymax = min(max((float(xml_box.find('ymax').text)), ymin), height)

            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
                label.append([xmin, ymin, xmax, ymax, cls_id, difficult])
            except AssertionError as e:
                logging.warning("Invalid label at %s, %s", anno_path, e)
        return np.array(label)
