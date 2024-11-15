# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class VOCDataset(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        # 'classes':
        # ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        #  'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        #  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        # # palette is a list of color tuples, which is used for visualization.
        # 'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
        #             (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
        #             (153, 69, 1), (120, 166, 157), (0, 182, 199),
        #             (0, 226, 252), (182, 182, 255), (0, 0, 230), (220, 20, 60),
        #             (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100),
        #             (183, 130, 88)]


        # # opixray 5类
        'classes': ('Folding_Knife', 'Straight_Knife', 'Scissor', 'Utility_Knife', 'Multi-tool_Knife'),
        'palette': [
            (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
            (106, 0, 228)
        ]

        # hixray 8类
        # 'classes': ('Portable_Charger_1', 'Portable_Charger_2', 'Water', 'Laptop', 'Mobile_Phone', 'Tablet',
        #             'Cosmetic', 'Nonmetallic_Lighter'),
        # 'palette': [
        #     (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
        #     (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
        # ]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None
