# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Tuple, Union

import torch
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import OptConfigType
from .deformable_detr_layers import DeformableDetrTransformerDecoder
from .utils import MLP, coordinate_to_encoding, inverse_sigmoid
from .dino_layers import CdnQueryGenerator


# class DinoTransformerDecoder(DeformableDetrTransformerDecoder):
#     """Transformer decoder of DINO."""
#
#     def _init_layers(self) -> None:
#         """Initialize decoder layers."""
#         super()._init_layers()
#         self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
#                                   self.embed_dims, 2)
#         self.norm = nn.LayerNorm(self.embed_dims)
#
#     def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
#                 self_attn_mask: Tensor, reference_points: Tensor,
#                 spatial_shapes: Tensor, level_start_index: Tensor,
#                 valid_ratios: Tensor, reg_branches: nn.ModuleList,
#                 **kwargs) -> Tuple[Tensor]:
#         """Forward function of Transformer decoder.
#
#         Args:
#             query (Tensor): The input query, has shape (num_queries, bs, dim).
#             value (Tensor): The input values, has shape (num_value, bs, dim).
#             key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
#                 input. ByteTensor, has shape (num_queries, bs).
#             self_attn_mask (Tensor): The attention mask to prevent information
#                 leakage from different denoising groups and matching parts, has
#                 shape (num_queries_total, num_queries_total). It is `None` when
#                 `self.training` is `False`.
#             reference_points (Tensor): The initial reference, has shape
#                 (bs, num_queries, 4) with the last dimension arranged as
#                 (cx, cy, w, h).
#             spatial_shapes (Tensor): Spatial shapes of features in all levels,
#                 has shape (num_levels, 2), last dimension represents (h, w).
#             level_start_index (Tensor): The start index of each level.
#                 A tensor has shape (num_levels, ) and can be represented
#                 as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
#             valid_ratios (Tensor): The ratios of the valid width and the valid
#                 height relative to the width and the height of features in all
#                 levels, has shape (bs, num_levels, 2).
#             reg_branches: (obj:`nn.ModuleList`): Used for refining the
#                 regression results.
#
#         Returns:
#             tuple[Tensor]: Output queries and references of Transformer
#                 decoder
#
#             - query (Tensor): Output embeddings of the last decoder, has
#               shape (num_queries, bs, embed_dims) when `return_intermediate`
#               is `False`. Otherwise, Intermediate output embeddings of all
#               decoder layers, has shape (num_decoder_layers, num_queries, bs,
#               embed_dims).
#             - reference_points (Tensor): The reference of the last decoder
#               layer, has shape (bs, num_queries, 4)  when `return_intermediate`
#               is `False`. Otherwise, Intermediate references of all decoder
#               layers, has shape (num_decoder_layers, bs, num_queries, 4). The
#               coordinates are arranged as (cx, cy, w, h)
#         """
#         intermediate = []
#         intermediate_reference_points = [reference_points]
#         for lid, layer in enumerate(self.layers):
#             if reference_points.shape[-1] == 4:
#                 reference_points_input = \
#                     reference_points[:, :, None] * torch.cat(
#                         [valid_ratios, valid_ratios], -1)[:, None]
#             else:
#                 assert reference_points.shape[-1] == 2
#                 reference_points_input = \
#                     reference_points[:, :, None] * valid_ratios[:, None]
#
#             query_sine_embed = coordinate_to_encoding(
#                 reference_points_input[:, :, 0, :])  # 这里是DAB中的PE方法，query_sine_embed是reference_point进行positional embeddings,（区别于encoding，这里使用每个位置的内容进行三角函数映射）
#             query_pos = self.ref_point_head(query_sine_embed)  # DAB中的PE方法，将xywh分别映射到128再cat到一起后用MLP映射到256维这样就能和object query进行相加
#
#             query = layer(
#                 query,
#                 query_pos=query_pos,
#                 value=value,
#                 key_padding_mask=key_padding_mask,
#                 self_attn_mask=self_attn_mask,
#                 spatial_shapes=spatial_shapes,
#                 level_start_index=level_start_index,
#                 valid_ratios=valid_ratios,
#                 reference_points=reference_points_input,
#                 **kwargs)
#
#             if reg_branches is not None:
#                 tmp = reg_branches[lid](query) # 用decoder输出的特征映射成偏移量tmp
#                 assert reference_points.shape[-1] == 4
#                 new_reference_points = tmp + inverse_sigmoid(
#                     reference_points, eps=1e-3)
#                 new_reference_points = new_reference_points.sigmoid()  # 这里能看出来似乎decoder的iterative部分的确只修正box而不再涉及到分类的问题上了。不过保存了query特征用于预测类别信息
#                 reference_points = new_reference_points.detach()  # TODO：这里涉及到detach需要再看看
#
#             if self.return_intermediate:  # 这里涉及到了Look Forward Twice
#                 intermediate.append(self.norm(query))  # intermediate只有6个并且存的是每层decoder输出的特征
#                 intermediate_reference_points.append(new_reference_points)  # 这个因为默认有一个reference points所以变成了7个
#                 # NOTE this is for the "Look Forward Twice" module,
#                 # in the DeformDETR, reference_points was appended.
#
#         if self.return_intermediate:
#             return torch.stack(intermediate), torch.stack(
#                 intermediate_reference_points)
#
#         return query, reference_points


class CdnQueryGeneratorv2(CdnQueryGenerator):
    """
    只是改变cdnquerygenrator中的属性label_embedding的定义方式，变成了直接从dinov2中传入，和match part的query共用labelembedding的参数
    """

    def __init__(self,
                 label_embedding,
                 num_classes: int,
                 embed_dims: int,
                 num_matching_queries: int,
                 label_noise_scale: float = 0.5,
                 box_noise_scale: float = 1.0,
                 group_cfg: OptConfigType = None) -> None:
        super().__init__(
            num_classes = num_classes,
            embed_dims = embed_dims,
            num_matching_queries=num_matching_queries,
            label_noise_scale=label_noise_scale,
            box_noise_scale=box_noise_scale,
            group_cfg=group_cfg

        )
        self.label_embedding = label_embedding  # 只是改变cdnquerygenrator中的属性label_embedding的定义方式，变成了直接从dinov2中传入，和match part的query共用labelembedding的参数
    def generate_dn_mask(self, max_num_target: int, num_groups: int,
                         device: Union[torch.device, str]) -> Tensor:
        """Generate attention mask to prevent information leakage from
        different denoising groups and matching parts.

        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.
        改写成：每个类别的种类都会进行
        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 1 1 1
                        1 1 1 1 1 1 1 1 0 0 1 1 1
                        1 1 1 1 1 1 1 1 1 1 0 0 1
                        1 1 1 1 1 1 1 1 1 1 0 0 1
                        1 1 1 1 1 1 1 1 1 1 1 1 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.
        Args:
            max_num_target (int): The max target number of the input batch
                samples.
            num_groups (int): The number of denoising query groups.
            device (obj:`device` or str): The device of generated mask.

        Returns:
            Tensor: The attention mask to prevent information leakage from
            different denoising groups and matching parts, will be used as
            `self_attn_mask` of the `decoder`, has shape (num_queries_total,
            num_queries_total), where `num_queries_total` is the sum of
            `num_denoising_queries` and `num_matching_queries`.
        """
        num_denoising_queries = int(max_num_target * 2 * num_groups)
        num_queries_total = num_denoising_queries + self.num_matching_queries
        attn_mask = torch.zeros(
            num_queries_total,
            num_queries_total,
            device=device,
            dtype=torch.bool)
        # Make the matching part cannot see the denoising groups
        attn_mask[num_denoising_queries:, :num_denoising_queries] = True
        # Make the denoising groups cannot see each other
        for i in range(num_groups):
            # Mask rows of one group per step.
            row_scope = slice(max_num_target * 2 * i,
                              max_num_target * 2 * (i + 1)) # 每组个数包含正负为二倍的max_num_gt_in_batch*2
            left_scope = slice(max_num_target * 2 * i)
            right_scope = slice(max_num_target * 2 * (i + 1),
                                num_denoising_queries)
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True
        # 让match部分的每个小组/类别互相之间不可见
        num = self.num_matching_queries//self.num_classes
        for c in range(self.num_classes):
            # Mask rows of one class group per step.
            row_scope = slice(num_denoising_queries+num * c,
                              num_denoising_queries+num * (c + 1))  # 当前类别组的纵向索引号
            left_scope = slice(num_denoising_queries,num_denoising_queries+num * c)  # 当前组左侧的索引范围
            right_scope = slice(num_denoising_queries+num * (c + 1),
                                num_queries_total)  # 当前组右侧的索引范围
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True


        return attn_mask

class CdnQueryGeneratorv2HaveEmbedding(CdnQueryGenerator):
    """
    只是改变cdnquerygenrator中的属性label_embedding的定义方式，变成了直接从dinov2中传入，和match part的query共用labelembedding的参数
    """

    def __init__(self,
                 num_classes: int,
                 embed_dims: int,
                 num_matching_queries: int,
                 label_noise_scale: float = 0.5,
                 box_noise_scale: float = 1.0,
                 group_cfg: OptConfigType = None) -> None:
        super().__init__(
            num_classes = num_classes,
            embed_dims = embed_dims,
            num_matching_queries=num_matching_queries,
            label_noise_scale=label_noise_scale,
            box_noise_scale=box_noise_scale,
            group_cfg=group_cfg

        )
        # self.label_embedding = None
        # self.label_embedding = label_embedding  # 只是改变cdnquerygenrator中的属性label_embedding的定义方式，变成了直接从dinov2中传入，和match part的query共用labelembedding的参数
    def generate_dn_mask(self, max_num_target: int, num_groups: int,
                         device: Union[torch.device, str]) -> Tensor:
        """Generate attention mask to prevent information leakage from
        different denoising groups and matching parts.

        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.
        改写成：每个类别的种类都会进行
        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 1 1 1
                        1 1 1 1 1 1 1 1 0 0 1 1 1
                        1 1 1 1 1 1 1 1 1 1 0 0 1
                        1 1 1 1 1 1 1 1 1 1 0 0 1
                        1 1 1 1 1 1 1 1 1 1 1 1 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.
        Args:
            max_num_target (int): The max target number of the input batch
                samples.
            num_groups (int): The number of denoising query groups.
            device (obj:`device` or str): The device of generated mask.

        Returns:
            Tensor: The attention mask to prevent information leakage from
            different denoising groups and matching parts, will be used as
            `self_attn_mask` of the `decoder`, has shape (num_queries_total,
            num_queries_total), where `num_queries_total` is the sum of
            `num_denoising_queries` and `num_matching_queries`.
        """
        num_denoising_queries = int(max_num_target * 2 * num_groups)
        num_queries_total = num_denoising_queries + self.num_matching_queries
        attn_mask = torch.zeros(
            num_queries_total,
            num_queries_total,
            device=device,
            dtype=torch.bool)
        # Make the matching part cannot see the denoising groups
        attn_mask[num_denoising_queries:, :num_denoising_queries] = True
        # Make the denoising groups cannot see each other
        for i in range(num_groups):
            # Mask rows of one group per step.
            row_scope = slice(max_num_target * 2 * i,
                              max_num_target * 2 * (i + 1)) # 每组个数包含正负为二倍的max_num_gt_in_batch*2
            left_scope = slice(max_num_target * 2 * i)
            right_scope = slice(max_num_target * 2 * (i + 1),
                                num_denoising_queries)
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True
        # 让match部分的每个小组/类别互相之间不可见
        num = self.num_matching_queries//self.num_classes
        for c in range(self.num_classes):
            # Mask rows of one class group per step.
            row_scope = slice(num_denoising_queries+num * c,
                              num_denoising_queries+num * (c + 1))  # 当前类别组的纵向索引号
            left_scope = slice(num_denoising_queries,num_denoising_queries+num * c)  # 当前组左侧的索引范围
            right_scope = slice(num_denoising_queries+num * (c + 1),
                                num_queries_total)  # 当前组右侧的索引范围
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True


        return attn_mask

class CdnQueryGeneratorv2HaveNoEmbedding(CdnQueryGenerator):
    """
    只是改变cdnquerygenrator中的属性label_embedding的定义方式，变成了直接从dinov2中传入，和match part的query共用labelembedding的参数
    """

    def __init__(self,
                 num_classes: int,
                 embed_dims: int,
                 num_matching_queries: int,
                 label_noise_scale: float = 0.5,
                 box_noise_scale: float = 1.0,
                 group_cfg: OptConfigType = None) -> None:
        super().__init__(
            num_classes = num_classes,
            embed_dims = embed_dims,
            num_matching_queries=num_matching_queries,
            label_noise_scale=label_noise_scale,
            box_noise_scale=box_noise_scale,
            group_cfg=group_cfg

        )
        # self.label_embedding = label_embedding  # 只是改变cdnquerygenrator中的属性label_embedding的定义方式，变成了直接从dinov2中传入，和match part的query共用labelembedding的参数
    def generate_dn_mask(self, max_num_target: int, num_groups: int,
                         device: Union[torch.device, str]) -> Tensor:
        """Generate attention mask to prevent information leakage from
        different denoising groups and matching parts.

        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.
        改写成：每个类别的种类都会进行
        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 1 1 1
                        1 1 1 1 1 1 1 1 0 0 1 1 1
                        1 1 1 1 1 1 1 1 1 1 0 0 1
                        1 1 1 1 1 1 1 1 1 1 0 0 1
                        1 1 1 1 1 1 1 1 1 1 1 1 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.
        Args:
            max_num_target (int): The max target number of the input batch
                samples.
            num_groups (int): The number of denoising query groups.
            device (obj:`device` or str): The device of generated mask.

        Returns:
            Tensor: The attention mask to prevent information leakage from
            different denoising groups and matching parts, will be used as
            `self_attn_mask` of the `decoder`, has shape (num_queries_total,
            num_queries_total), where `num_queries_total` is the sum of
            `num_denoising_queries` and `num_matching_queries`.
        """
        num_denoising_queries = int(max_num_target * 2 * num_groups)
        num_queries_total = num_denoising_queries + self.num_matching_queries
        attn_mask = torch.zeros(
            num_queries_total,
            num_queries_total,
            device=device,
            dtype=torch.bool)
        # Make the matching part cannot see the denoising groups
        attn_mask[num_denoising_queries:, :num_denoising_queries] = True
        # Make the denoising groups cannot see each other
        for i in range(num_groups):
            # Mask rows of one group per step.
            row_scope = slice(max_num_target * 2 * i,
                              max_num_target * 2 * (i + 1)) # 每组个数包含正负为二倍的max_num_gt_in_batch*2
            left_scope = slice(max_num_target * 2 * i)
            right_scope = slice(max_num_target * 2 * (i + 1),
                                num_denoising_queries)
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True
        # 让match部分的每个小组/类别互相之间不可见
        num = self.num_matching_queries//self.num_classes
        for c in range(self.num_classes):
            # Mask rows of one class group per step.
            row_scope = slice(num_denoising_queries+num * c,
                              num_denoising_queries+num * (c + 1))  # 当前类别组的纵向索引号
            left_scope = slice(num_denoising_queries,num_denoising_queries+num * c)  # 当前组左侧的索引范围
            right_scope = slice(num_denoising_queries+num * (c + 1),
                                num_queries_total)  # 当前组右侧的索引范围
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True


        return attn_mask
class CdnQueryGeneratorv2UpperRightCorner(CdnQueryGenerator):
    """
    从实验3看出来使用cdnquerygenrator中的属性label_embedding传给match query的方式进行共享性能下降最小，可能是因为这样会从预训练模型中加载
    因此本模块不进行label embedding的额外定义，因为会自动继承
    只负责mask掩码的定义，这个模块负责让右上角不被掩掉
    """

    def __init__(self,
                 num_classes: int,
                 embed_dims: int,
                 num_matching_queries: int,
                 label_noise_scale: float = 0.5,
                 box_noise_scale: float = 1.0,
                 group_cfg: OptConfigType = None) -> None:
        super().__init__(
            num_classes = num_classes,
            embed_dims = embed_dims,
            num_matching_queries=num_matching_queries,
            label_noise_scale=label_noise_scale,
            box_noise_scale=box_noise_scale,
            group_cfg=group_cfg

        )
        # self.label_embedding = label_embedding  # 只是改变cdnquerygenrator中的属性label_embedding的定义方式，变成了直接从dinov2中传入，和match part的query共用labelembedding的参数
    def generate_dn_mask(self, max_num_target: int, num_groups: int,
                         device: Union[torch.device, str]) -> Tensor:
        """Generate attention mask to prevent information leakage from
        different denoising groups and matching parts.

        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.
        改写成：每个类别的种类都会进行
        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 1 1 1
                        1 1 1 1 1 1 1 1 0 0 1 1 1
                        1 1 1 1 1 1 1 1 1 1 0 0 1
                        1 1 1 1 1 1 1 1 1 1 0 0 1
                        1 1 1 1 1 1 1 1 1 1 1 1 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.
        Args:
            max_num_target (int): The max target number of the input batch
                samples.
            num_groups (int): The number of denoising query groups.
            device (obj:`device` or str): The device of generated mask.

        Returns:
            Tensor: The attention mask to prevent information leakage from
            different denoising groups and matching parts, will be used as
            `self_attn_mask` of the `decoder`, has shape (num_queries_total,
            num_queries_total), where `num_queries_total` is the sum of
            `num_denoising_queries` and `num_matching_queries`.
        """
        num_denoising_queries = int(max_num_target * 2 * num_groups)
        num_queries_total = num_denoising_queries + self.num_matching_queries
        attn_mask = torch.zeros(
            num_queries_total,
            num_queries_total,
            device=device,
            dtype=torch.bool)
        # Make the matching part cannot see the denoising groups
        attn_mask[num_denoising_queries:, :num_denoising_queries] = True
        # Make the denoising groups cannot see each other
        for i in range(num_groups):
            # Mask rows of one group per step.
            row_scope = slice(max_num_target * 2 * i,
                              max_num_target * 2 * (i + 1)) # 每组个数包含正负为二倍的max_num_gt_in_batch*2
            left_scope = slice(max_num_target * 2 * i)
            right_scope = slice(max_num_target * 2 * (i + 1),
                                num_denoising_queries)
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True
        # 让match部分的每个小组/类别互相之间不可见
        num = self.num_matching_queries//self.num_classes
        for c in range(self.num_classes):
            # Mask rows of one class group per step.
            row_scope = slice(num_denoising_queries+num * c,
                              num_denoising_queries+num * (c + 1))  # 当前类别组的纵向索引号
            left_scope = slice(num_denoising_queries,num_denoising_queries+num * c)  # 当前组左侧的索引范围
            # right_scope = slice(num_denoising_queries+num * (c + 1),
            #                     num_queries_total)  # 当前组右侧的索引范围
            # attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True


        return attn_mask
class CdnQueryGeneratorv2LowerLeftCorner(CdnQueryGenerator):
    """
    从实验3看出来使用cdnquerygenrator中的属性label_embedding传给match query的方式进行共享性能下降最小，可能是因为这样会从预训练模型中加载
    因此本模块不进行label embedding的额外定义，因为会自动继承
    只负责mask掩码的定义，这个模块负责让左下角不被掩掉
    """

    def __init__(self,
                 num_classes: int,
                 embed_dims: int,
                 num_matching_queries: int,
                 label_noise_scale: float = 0.5,
                 box_noise_scale: float = 1.0,
                 group_cfg: OptConfigType = None) -> None:
        super().__init__(
            num_classes = num_classes,
            embed_dims = embed_dims,
            num_matching_queries=num_matching_queries,
            label_noise_scale=label_noise_scale,
            box_noise_scale=box_noise_scale,
            group_cfg=group_cfg

        )
        # self.label_embedding = label_embedding  # 只是改变cdnquerygenrator中的属性label_embedding的定义方式，变成了直接从dinov2中传入，和match part的query共用labelembedding的参数
    def generate_dn_mask(self, max_num_target: int, num_groups: int,
                         device: Union[torch.device, str]) -> Tensor:
        """Generate attention mask to prevent information leakage from
        different denoising groups and matching parts.

        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.
        改写成：每个类别的种类都会进行
        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 1 1 1
                        1 1 1 1 1 1 1 1 0 0 1 1 1
                        1 1 1 1 1 1 1 1 1 1 0 0 1
                        1 1 1 1 1 1 1 1 1 1 0 0 1
                        1 1 1 1 1 1 1 1 1 1 1 1 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.
        Args:
            max_num_target (int): The max target number of the input batch
                samples.
            num_groups (int): The number of denoising query groups.
            device (obj:`device` or str): The device of generated mask.

        Returns:
            Tensor: The attention mask to prevent information leakage from
            different denoising groups and matching parts, will be used as
            `self_attn_mask` of the `decoder`, has shape (num_queries_total,
            num_queries_total), where `num_queries_total` is the sum of
            `num_denoising_queries` and `num_matching_queries`.
        """
        num_denoising_queries = int(max_num_target * 2 * num_groups)
        num_queries_total = num_denoising_queries + self.num_matching_queries
        attn_mask = torch.zeros(
            num_queries_total,
            num_queries_total,
            device=device,
            dtype=torch.bool)
        # Make the matching part cannot see the denoising groups
        attn_mask[num_denoising_queries:, :num_denoising_queries] = True
        # Make the denoising groups cannot see each other
        for i in range(num_groups):
            # Mask rows of one group per step.
            row_scope = slice(max_num_target * 2 * i,
                              max_num_target * 2 * (i + 1)) # 每组个数包含正负为二倍的max_num_gt_in_batch*2
            left_scope = slice(max_num_target * 2 * i)
            right_scope = slice(max_num_target * 2 * (i + 1),
                                num_denoising_queries)
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True
        # 让match部分的每个小组/类别互相之间不可见
        num = self.num_matching_queries//self.num_classes
        for c in range(self.num_classes):
            # Mask rows of one class group per step.
            row_scope = slice(num_denoising_queries+num * c,
                              num_denoising_queries+num * (c + 1))  # 当前类别组的纵向索引号
            # left_scope = slice(num_denoising_queries,num_denoising_queries+num * c)  # 当前组左侧的索引范围
            right_scope = slice(num_denoising_queries+num * (c + 1),
                                num_queries_total)  # 当前组右侧的索引范围
            attn_mask[row_scope, right_scope] = True
            # attn_mask[row_scope, left_scope] = True


        return attn_mask
class CdnQueryGeneratorv2ALL0(CdnQueryGenerator):
    """
    从实验3看出来使用cdnquerygenrator中的属性label_embedding传给match query的方式进行共享性能下降最小，可能是因为这样会从预训练模型中加载
    因此本模块不进行label embedding的额外定义，因为会自动继承
    只负责mask掩码的定义，这个模块负责让左下角不被掩掉
    """

    def __init__(self,
                 num_classes: int,
                 embed_dims: int,
                 num_matching_queries: int,
                 label_noise_scale: float = 0.5,
                 box_noise_scale: float = 1.0,
                 group_cfg: OptConfigType = None) -> None:
        super().__init__(
            num_classes = num_classes,
            embed_dims = embed_dims,
            num_matching_queries=num_matching_queries,
            label_noise_scale=label_noise_scale,
            box_noise_scale=box_noise_scale,
            group_cfg=group_cfg

        )
        # self.label_embedding = label_embedding  # 只是改变cdnquerygenrator中的属性label_embedding的定义方式，变成了直接从dinov2中传入，和match part的query共用labelembedding的参数
    def generate_dn_mask(self, max_num_target: int, num_groups: int,
                         device: Union[torch.device, str]) -> Tensor:
        """Generate attention mask to prevent information leakage from
        different denoising groups and matching parts.

        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.
        改写成：每个类别的种类都会进行
        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 1 1 1
                        1 1 1 1 1 1 1 1 0 0 1 1 1
                        1 1 1 1 1 1 1 1 1 1 0 0 1
                        1 1 1 1 1 1 1 1 1 1 0 0 1
                        1 1 1 1 1 1 1 1 1 1 1 1 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.
        Args:
            max_num_target (int): The max target number of the input batch
                samples.
            num_groups (int): The number of denoising query groups.
            device (obj:`device` or str): The device of generated mask.

        Returns:
            Tensor: The attention mask to prevent information leakage from
            different denoising groups and matching parts, will be used as
            `self_attn_mask` of the `decoder`, has shape (num_queries_total,
            num_queries_total), where `num_queries_total` is the sum of
            `num_denoising_queries` and `num_matching_queries`.
        """
        num_denoising_queries = int(max_num_target * 2 * num_groups)
        num_queries_total = num_denoising_queries + self.num_matching_queries
        attn_mask = torch.zeros(
            num_queries_total,
            num_queries_total,
            device=device,
            dtype=torch.bool)


        return attn_mask