# Copyright (c) OpenMMLab. All rights reserved.
from .conditional_detr_layers import (ConditionalDetrTransformerDecoder,
                                      ConditionalDetrTransformerDecoderLayer)
from .dab_detr_layers import (DABDetrTransformerDecoder,
                              DABDetrTransformerDecoderLayer,
                              DABDetrTransformerEncoder)
from .deformable_detr_layers import (DeformableDetrTransformerDecoder,
                                     DeformableDetrTransformerDecoderLayer,
                                     DeformableDetrTransformerEncoder,
                                     DeformableDetrTransformerEncoderLayer)
from .detr_layers import (DetrTransformerDecoder, DetrTransformerDecoderLayer,
                          DetrTransformerEncoder, DetrTransformerEncoderLayer)
# from .dino_layers import CdnQueryGenerator, DinoTransformerDecoder  # DINO 默认的DinoTransformerDecoder意味着不启用 LFN
from .mask2former_layers import (Mask2FormerTransformerDecoder,
                                 Mask2FormerTransformerDecoderLayer,
                                 Mask2FormerTransformerEncoder)
from .utils import (MLP, AdaptivePadding, ConditionalAttention, DynamicConv,
                    PatchEmbed, PatchMerging, coordinate_to_encoding,
                    inverse_sigmoid, nchw_to_nlc, nlc_to_nchw)
from .dinov2_layers import CdnQueryGeneratorv2,CdnQueryGeneratorv2HaveEmbedding # 改变label embedding的来源

from .dinov2_six_layers_average import CdnQueryGenerator, DinoTransformerDecoder  # AO-DETR 的DinoTransformerDecoder意味着启用 LFN

from .dinov2_layers import CdnQueryGeneratorv2LowerLeftCorner,CdnQueryGeneratorv2UpperRightCorner,CdnQueryGeneratorv2ALL0

__all__ = [
    'nlc_to_nchw', 'nchw_to_nlc', 'AdaptivePadding', 'PatchEmbed',
    'PatchMerging', 'inverse_sigmoid', 'DynamicConv', 'MLP',
    'DetrTransformerEncoder', 'DetrTransformerDecoder',
    'DetrTransformerEncoderLayer', 'DetrTransformerDecoderLayer',
    'DeformableDetrTransformerEncoder', 'DeformableDetrTransformerDecoder',
    'DeformableDetrTransformerEncoderLayer',
    'DeformableDetrTransformerDecoderLayer', 'coordinate_to_encoding',
    'ConditionalAttention', 'DABDetrTransformerDecoderLayer',
    'DABDetrTransformerDecoder', 'DABDetrTransformerEncoder',
    'ConditionalDetrTransformerDecoder',
    'ConditionalDetrTransformerDecoderLayer', 'DinoTransformerDecoder',
    'CdnQueryGenerator', 'Mask2FormerTransformerEncoder',
    'Mask2FormerTransformerDecoderLayer', 'Mask2FormerTransformerDecoder',
    'CdnQueryGeneratorv2','CdnQueryGeneratorv2HaveEmbedding',
    'CdnQueryGeneratorv2LowerLeftCorner','CdnQueryGeneratorv2UpperRightCorner',
    'CdnQueryGeneratorv2ALL0'
]
