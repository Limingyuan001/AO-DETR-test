# Copyright (c) OpenMMLab. All rights reserved.
from .anchor_free_head import AnchorFreeHead
from .anchor_head import AnchorHead
from .atss_head import ATSSHead
from .atss_vlfusion_head import ATSSVLFusionHead
from .autoassign_head import AutoAssignHead
from .boxinst_head import BoxInstBboxHead, BoxInstMaskHead
from .cascade_rpn_head import CascadeRPNHead, StageCascadeRPNHead
from .centernet_head import CenterNetHead
from .centernet_update_head import CenterNetUpdateHead
from .centripetal_head import CentripetalHead
from .condinst_head import CondInstBboxHead, CondInstMaskHead
from .conditional_detr_head import ConditionalDETRHead
from .corner_head import CornerHead
from .dab_detr_head import DABDETRHead
from .ddod_head import DDODHead
from .deformable_detr_head import DeformableDETRHead
from .detr_head import DETRHead
from .dino_head import DINOHead
from .embedding_rpn_head import EmbeddingRPNHead
from .fcos_head import FCOSHead
from .fovea_head import FoveaHead
from .free_anchor_retina_head import FreeAnchorRetinaHead
from .fsaf_head import FSAFHead
from .ga_retina_head import GARetinaHead
from .ga_rpn_head import GARPNHead
from .gfl_head import GFLHead
from .guided_anchor_head import FeatureAdaption, GuidedAnchorHead
from .lad_head import LADHead
from .ld_head import LDHead
from .mask2former_head import Mask2FormerHead
from .maskformer_head import MaskFormerHead
from .nasfcos_head import NASFCOSHead
from .paa_head import PAAHead
from .pisa_retinanet_head import PISARetinaHead
from .pisa_ssd_head import PISASSDHead
from .reppoints_head import RepPointsHead
from .retina_head import RetinaHead
from .retina_sepbn_head import RetinaSepBNHead
from .rpn_head import RPNHead
from .rtmdet_head import RTMDetHead, RTMDetSepBNHead
from .rtmdet_ins_head import RTMDetInsHead, RTMDetInsSepBNHead
from .sabl_retina_head import SABLRetinaHead
from .solo_head import DecoupledSOLOHead, DecoupledSOLOLightHead, SOLOHead
from .solov2_head import SOLOV2Head
from .ssd_head import SSDHead
from .tood_head import TOODHead
from .vfnet_head import VFNetHead
from .yolact_head import YOLACTHead, YOLACTProtonet
from .yolo_head import YOLOV3Head
from .yolof_head import YOLOFHead
from .yolox_head import YOLOXHead

# from .dinov2_head import DINOHeadv2
# from .dinov2_head_2 import DINOHeadv2
# from .dinov2_head_3 import DINOHeadv2
# from .dinov2_head_4 import DINOHeadv2
# from .dinov2_head_5 import DINOHeadv2
from .dinov2_head_6 import DINOHeadv2

# from .dinov3_head import DINOHeadv3  # 'DINOHeadv3' 用于AO-DETRv2 # time 3/20 2024
# from .dinov3_head_2 import DINOHeadv3  # 'DINOHeadv3' 用于AO-DETRv2 # time 3/20 2024
# from .dinov3_head_3 import DINOHeadv3  # 'DINOHeadv3' 用于AO-DETRv2 # time 3/20 2024 高效掩码
from .dinov3_head_4 import DINOHeadv3  # 'DINOHeadv3' 用于AO-DETRv2 # time 3/20 2024 进一步对simmax类别内部的对角线掩码 使用的是0.01阈值掩码，基本好用，但是偶尔出nan
# from .dinov3_head_5 import DINOHeadv3  # 'DINOHeadv3' 用于AO-DETRv2 # time 4/7 2024 进一步对simmax类别内部的对角线掩码，使用maskeye进行掩码，不会出nan但是loss会变小
# from .dinov3_head_6 import DINOHeadv3# 'DINOHeadv3' 用于AO-DETRv2 在AO-DETR的dinov2head6上添加对比学习# time 4/13 2024
# from .dinov3_head_7 import DINOHeadv3# 'DINOHeadv3' 用于AO-DETRv2 在AO-DETR的dinov2head6基础上取消rank-weight time 4/14 2024
# from .dinov3_head_infoNCE2 import DINOHeadv3


from .deformable_detr_head_4 import DeformableDETRHeadv3  # 用于AO-DETRv2的泛化性 time 3/26 2024 为了测试泛化性，在deform-detr的head上进行移植
from .dab_detr_head_4 import DABDETRHeadv3  # 用于AO-DETRv2的泛化性 time 3/27 2024 为了测试泛化性，在dab-detr的head上进行移植


# from .dinov4_head_4 import DINOHeadv4
# from .dinov4_head_5 import DINOHeadv4
# from .dinov4_head_6 import DINOHeadv4
# from .dinov4_head_7 import DINOHeadv4
# from .dinov4_head_8 import DINOHeadv4
# from .dinov4_head_9 import DINOHeadv4
# from .dinov4_head_10 import DINOHeadv4
# from .dinov4_head_11 import DINOHeadv4
from .dinov4_head_12 import DINOHeadv4  # CSPCL 最佳184 dino 67.5 layer all
# from .dinov4_head_infoNCE import DINOHeadv4
# from .dinov4_head_infoNCE2 import DINOHeadv4
from .ao_detrv4_head_12 import AODETRHeadv4

# from .deformable_detrv4_head_10 import DeformableDETRHeadv4  # 用于MMCLv2的泛化性 time 9/10 2024 为了测试泛化性，在deform-detr的head上进行移植
from .deformable_detrv4_head_12 import DeformableDETRHeadv4


__all__ = [
    'AnchorFreeHead', 'AnchorHead', 'GuidedAnchorHead', 'FeatureAdaption',
    'RPNHead', 'GARPNHead', 'RetinaHead', 'RetinaSepBNHead', 'GARetinaHead',
    'SSDHead', 'FCOSHead', 'RepPointsHead', 'FoveaHead',
    'FreeAnchorRetinaHead', 'ATSSHead', 'FSAFHead', 'NASFCOSHead',
    'PISARetinaHead', 'PISASSDHead', 'GFLHead', 'CornerHead', 'YOLACTHead',
    'YOLACTProtonet', 'YOLOV3Head', 'PAAHead', 'SABLRetinaHead',
    'CentripetalHead', 'VFNetHead', 'StageCascadeRPNHead', 'CascadeRPNHead',
    'EmbeddingRPNHead', 'LDHead', 'AutoAssignHead', 'DETRHead', 'YOLOFHead',
    'DeformableDETRHead', 'CenterNetHead', 'YOLOXHead', 'SOLOHead',
    'DecoupledSOLOHead', 'DecoupledSOLOLightHead', 'SOLOV2Head', 'LADHead',
    'TOODHead', 'MaskFormerHead', 'Mask2FormerHead', 'DDODHead',
    'CenterNetUpdateHead', 'RTMDetHead', 'RTMDetSepBNHead', 'CondInstBboxHead',
    'CondInstMaskHead', 'RTMDetInsHead', 'RTMDetInsSepBNHead',
    'BoxInstBboxHead', 'BoxInstMaskHead', 'ConditionalDETRHead', 'DINOHead',
    'ATSSVLFusionHead', 'DABDETRHead',
    'DINOHeadv2',
    'DINOHeadv3',
    'DeformableDETRHeadv3',
    'DABDETRHeadv3',
    'DINOHeadv4',
    'DeformableDETRHeadv4',
    'AODETRHeadv4'
]
