# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from mmengine.structures import InstanceData
from torch import Tensor
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, bbox_overlaps
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from ..utils import multi_apply
from .deformable_detr_head import DeformableDETRHead
from .dino_head import DINOHead

from typing import Optional
from ..task_modules.assigners.assign_result import AssignResult
from scipy.optimize import linear_sum_assignment

# TODO AO-DETR是基于dinov2_head6的基础上的，为了引入dinov4head12中的cspcl机制，我们建ao_detrv4head_12.py 10/18/2024
import json
import os
import torch.nn as nn
import torch.nn.functional as F
# 提供的 cos_sim 函数
def cos_sim(embedded):
    embedded = F.normalize(embedded, dim=2)
    sim = torch.matmul(embedded, embedded.transpose(1, 2))
    return torch.clamp(sim, min=0.0005, max=0.9995)



# SimMinLoss 类
class SimMinLossv2(nn.Module):
    def __init__(self, margin=0.15, metric='cos', reduction='mean', num_queries=30, cls_weight=None):
        super(SimMinLossv2, self).__init__()
        self.m = margin
        self.metric = metric
        self.reduction = reduction
        # done 优于queries数量不等于类别数量，这里加入相似矩阵的宽高信息，也就是queries的数量，后续类别原型矩阵进行扩展
        self.num_queries = num_queries
        self.cls_weight = cls_weight

    def create_mask(self, size, class_num):
        """
        创建一个掩码，其中对角线上的 NxN 正方形区域是 False，其他是 True。
        :param size: 总共有多少个向量 M
        :param N: 对角线上每个正方形的边长
        :return: 掩码 Tensor
        """
        mask = torch.ones((size, size), dtype=torch.bool)
        N = size//class_num
        for i in range(0, size, N):
            if (i + N) <= size:  # 确保不超出边界
                mask[i:i + N, i:i + N] = False
        return mask

    # def create_mask(self, size, class_num):
    #     # Calculate the size of the small square (N)
    #     N = size // class_num
    #     # Create a mask for a single class
    #     small_mask = torch.ones((N, N), dtype=torch.bool)
    #     eye = torch.eye(N, dtype=torch.bool)
    #     # Use broadcasting to create the diagonal pattern
    #     mask = small_mask.unsqueeze(0) - eye.unsqueeze(1)
    #     # Expand the pattern to the full mask size
    #     full_mask = mask.repeat(class_num, class_num, 1, 1)
    #     # Reshape to get the final mask
    #     final_mask = full_mask.transpose(1, 2).reshape(size, size)
    #     return final_mask

    def forward(self, embedded, class_num):
        B, M, C = embedded.size()
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':

            # 创建掩码并应用到相似度矩阵
            mask = self.create_mask(M, class_num).to(embedded.device)
            mask = mask.unsqueeze(0).expand(B, M, M)  # 扩展掩码到批次大小

            # todo 使用分类器权重和content queries：embedded进行相似度计算。最后使用分类器类别相似度进行加权 20249/12
            # 首先对分类器参数进行归一化 15，256
            classifier = F.normalize(self.cls_weight, p=2, dim=1)  # classifier已经经过norm了，其shape: (15, 256)
            # 对分类器矩阵扩张到 batch,num_queries,256
            repeat_num = self.num_queries // classifier.size()[0]
            classifier = torch.repeat_interleave(classifier, repeats=repeat_num, dim=0).unsqueeze(
                0)  # 复制到queries个数，再扩展B维度
            classifier = torch.repeat_interleave(classifier, repeats=B, dim=0)  # 扩展B通道数量为batch


            # todo done 对embedded进行归一化
            content_queries = F.normalize(embedded, p=2, dim=-1)

            # 计算相似度矩阵，使用矩阵乘法实现
            similarity_matrix = torch.matmul(classifier, torch.transpose(content_queries,1,2))
            similarity_matrix = torch.clamp(similarity_matrix, min=0.0005, max=0.9995)

            # 对类内损失进行掩码操作
            usm = similarity_matrix.masked_fill(~mask, 0)  # 应用掩码upsampled_similarity_matrix

            # todo 使用分类器之间的相似度来对类间特征相似度损失进行加权，使得分布较为近距离的样本之间的样本对的loss变大 2024/9/11
            # 计算相似度矩阵，使用矩阵乘法实现
            similarity_matrix_ww = torch.matmul(classifier, torch.transpose(classifier, 1, 2))
            similarity_matrix_ww = torch.clamp(similarity_matrix_ww, min=0.0005, max=0.9995)

            # margin=torch.exp(torch.pow(similarity_matrix_ww,3))
            margin = torch.exp(similarity_matrix_ww*0.3+0.7)
            loss = -torch.log(1 - usm)*margin

            loss = loss.masked_select(mask)  # 仅选择掩码中为 True 的损失值
        else:
            raise NotImplementedError

        # 根据 reduction 参数进行损失的减少操作
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


# # 使用示例
# B, M, C = 2, 30, 5  # 示例批次大小，M 的数量和特征维度 C
# embedded = torch.randn(B, M, C)  # 示例输入
# sim_min_loss = SimMinLoss(metric='cos', reduction='mean')  # 实例化损失函数
# loss = sim_min_loss(embedded, 15)  # 计算损失
# print(loss)  # 输出损失值

# todo SimMaxLoss进行大改，直接让所有样本靠近分类器原型。
class SimMaxLossv2(nn.Module):
    def __init__(self, metric='cos', alpha=2.5, reduction='mean', num_queries=30, cls_weight=None):
        super(SimMaxLossv2, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

        self.num_queries = num_queries
        self.cls_weight = cls_weight
    def create_mask(self, size, class_num):
        """
        创建一个掩码，其中对角线上的 NxN 正方形区域是 False，其他是 True。
        :param size: 总共有多少个向量 M
        :param N: 对角线上每个正方形的边长
        :return: 掩码 Tensor
        """
        mask = torch.ones((size, size), dtype=torch.bool)
        N = size//class_num
        for i in range(0, size, N):
            if (i + N) <= size:  # 确保不超出边界
                mask[i:i + N, i:i + N] = False
        return mask
    def forward(self, embedded, class_num):
        """
        :param embedded: [B, M, C]
        :return:
        """
        B, M, C = embedded.size()
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            # # 使用之前定义的 cos_sim 函数计算相似度
            # sim = cos_sim(embedded)

            # todo mmcl 类间原始方案是利用样本对之间的相似度矩阵，现在改成样和分类头之间的相似度矩阵24/9/3
            normalized_weight = F.normalize(self.cls_weight, p=2, dim=-1)  # shape: (15, 256)

            # 计算相似度矩阵，使用矩阵乘法实现
            repeat_num = self.num_queries // normalized_weight.size()[0]
            normalized_weight = torch.repeat_interleave(normalized_weight, repeats=repeat_num, dim=0).unsqueeze(0)  # 复制到queries个数，再扩展B维度
            normalized_weight = torch.repeat_interleave(normalized_weight, repeats=B, dim=0)  # 扩展B通道数量为batch

            embedded_norm = F.normalize(embedded, p=2, dim=-1)

            similarity_matrix = torch.matmul(embedded_norm, torch.transpose(normalized_weight,1,2))
            # similarity_matrix = torch.mm(normalized_weight, normalized_weight.t())  # shape: (15, 15)
            sim = torch.clamp(similarity_matrix, min=0.0005, max=0.9995)  # 默认0.9995

            # 应用一个掩码将对角线上的方块的相似度进行保留，其他位置置零，这样能将继续计算类内相似度排名
            mask = self.create_mask(M, class_num).to(embedded.device)
            mask = mask.unsqueeze(0).expand(B, M, M)  # 扩展掩码到批次大小
            sim = sim.masked_fill(mask, 0.00001)  # 应用掩码
            loss = -torch.log(sim)


            # # 计算排名权重
            # _, indices = sim.sort(descending=True, dim=2)
            # _, rank = indices.sort(dim=2)
            # rank_weights = torch.exp(-rank.float() * self.alpha)
            #
            # # todo 试试要不要对rank进行mask, done 发现没有区别
            # rank_weights=rank_weights.masked_fill_(mask, 0)
            # # loss1=loss*rank_weights1
            # # loss1 = loss1[loss1 > 0]
            # # 应用排名权重
            # loss = loss * rank_weights
            #
            # # 过滤掉对角线上及负损失值
            # # loss = loss[loss > 0.01]  # todo 默认是m=0.01最好，但是实验需要调参 5/5/2024
            # loss = loss[loss > 0.01]

        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
# def store_data(img_meta, assign_result, layer_count=6, save_path='path/to/save'):
#     # 解析图片路径以获取图片名称
#     img_path = img_meta['img_path']
#     img_name = os.path.basename(img_path).split('.')[0]
#
#     # 为每张图片创建一个单独的JSON文件
#     save_filename = os.path.join(save_path, f'{img_name}.json')
#
#     # 尝试读取该图片的现有数据
#     img_data = {'layer_counts': [0] * layer_count, 'data': []}
#     if os.path.exists(save_filename):
#         try:
#             with open(save_filename, 'r') as file:
#                 img_data = json.load(file)
#         except json.JSONDecodeError:
#             print(f"Warning: Error reading {save_filename}. Starting fresh.")
#
#     # 确定当前的轮次和层
#     total_count = sum(img_data['layer_counts'])
#     current_epoch = total_count // layer_count + 1
#     current_layer = total_count % layer_count + 1
#
#     # 更新层数计数
#     img_data['layer_counts'][current_layer - 1] += 1
#
#     # 添加当前数据
#     current_data = {
#         'epoch': current_epoch,
#         'layer': current_layer,
#         'assign_result': assign_result.cpu().tolist()
#     }
#     img_data['data'].append(current_data)
#
#     # 写回该图片的更新后的数据
#     try:
#         with open(save_filename, 'w') as file:
#             json.dump(img_data, file)
#     except IOError as e:
#         print(f"Error writing to {save_filename}: {e}")
def store_data(img_meta, assign_result, layer_count=6, save_path='path/to/save'):
    # 解析图片路径以获取图片名称
    img_path = img_meta['img_path']
    img_name = os.path.basename(img_path).split('.')[0]

    # 为每张图片创建一个单独的JSON文件
    save_filename = os.path.join(save_path, f'{img_name}.json')

    # 尝试读取该图片的现有数据
    img_data = {'layer_counts': [0] * layer_count, 'data': []}
    if os.path.exists(save_filename):
        try:
            with open(save_filename, 'r') as file:
                img_data = json.load(file)
        except json.JSONDecodeError:
            print(f"Warning: Error reading {save_filename}. Starting fresh.")

    # 确定当前的轮次和层
    total_count = sum(img_data['layer_counts'])
    current_epoch = total_count // layer_count + 1
    current_layer = total_count % layer_count + 1

    # 更新层数计数
    img_data['layer_counts'][current_layer - 1] += 1

    # 添加当前数据
    current_data = {
        'epoch': current_epoch,
        'layer': current_layer,
        # 'assign_result': assign_result.cpu().tolist()
        'assign_result': assign_result.gt_inds.cpu().tolist(),
        'assign_result_labels': assign_result.labels.cpu().tolist()
        # .gt_inds, assign_result.labels,
    }
    img_data['data'].append(current_data)

    # 写回该图片的更新后的数据
    try:
        with open(save_filename, 'w') as file:
            json.dump(img_data, file)
    except IOError as e:
        print(f"Error writing to {save_filename}: {e}")

@MODELS.register_module()
class AODETRHeadv4(DINOHead):
    r"""Head of the DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2203.03605>`_ .
    """

    # def loss(self, hidden_states: Tensor, references: List[Tensor],
    #          enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
    #          batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:
    #     """Perform forward propagation and loss calculation of the detection
    #     head on the queries of the upstream network.
    #
    #     Args:
    #         hidden_states (Tensor): Hidden states output from each decoder
    #             layer, has shape (num_decoder_layers, bs, num_queries_total,
    #             dim), where `num_queries_total` is the sum of
    #             `num_denoising_queries` and `num_matching_queries` when
    #             `self.training` is `True`, else `num_matching_queries`.
    #         references (list[Tensor]): List of the reference from the decoder.
    #             The first reference is the `init_reference` (initial) and the
    #             other num_decoder_layers(6) references are `inter_references`
    #             (intermediate). The `init_reference` has shape (bs,
    #             num_queries_total, 4) and each `inter_reference` has shape
    #             (bs, num_queries, 4) with the last dimension arranged as
    #             (cx, cy, w, h).
    #         enc_outputs_class (Tensor): The score of each point on encode
    #             feature map, has shape (bs, num_feat_points, cls_out_channels).
    #         enc_outputs_coord (Tensor): The proposal generate from the
    #             encode feature map, has shape (bs, num_feat_points, 4) with the
    #             last dimension arranged as (cx, cy, w, h).
    #         batch_data_samples (list[:obj:`DetDataSample`]): The Data
    #             Samples. It usually includes information such as
    #             `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
    #         dn_meta (Dict[str, int]): The dictionary saves information about
    #           group collation, including 'num_denoising_queries' and
    #           'num_denoising_groups'. It will be used for split outputs of
    #           denoising and matching parts and loss calculation.
    #
    #     Returns:
    #         dict: A dictionary of loss components.
    #     """
    #     batch_gt_instances = []
    #     batch_img_metas = []
    #     for data_sample in batch_data_samples:
    #         batch_img_metas.append(data_sample.metainfo)
    #         batch_gt_instances.append(data_sample.gt_instances)
    #
    #     outs = self(hidden_states, references) # 这里对decoder的输出进行LFT处理，得到每层的预测结果
    #     loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
    #                           batch_gt_instances, batch_img_metas, dn_meta)  # 这里将decoder的预测结果和encoder筛选后的预测结果都放在了一起，还有真值，dn信息等
    #     losses = self.loss_by_feat(*loss_inputs)
    #     return losses
    # todo 上边本来就是被注释掉的，因为dino-head中的loss就被改了，
    #  所以想要加入对比学习的话，我们需要复制dinohead中的loss并且进行添加功能
    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries_total,
                dim), where `num_queries_total` is the sum of
                `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries_total, 4) and each `inter_reference` has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_outputs_coord (Tensor): The proposal generate from the
                encode feature map, has shape (bs, num_feat_points, 4) with the
                last dimension arranged as (cx, cy, w, h).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references) # 这里对decoder的输出进行LFT处理，得到每层的预测结果
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)  # 这里将decoder的预测结果和encoder筛选后的预测结果都放在了一起，还有真值，dn信息等
        losses = self.loss_by_feat(*loss_inputs)
        # todo 类间使用原本mmcl，类内使用新版特征和原型相似度
        sim_min_loss = SimMinLossv2(num_queries=self.num_queries,
                                    cls_weight=self.cls_branches[-1].weight)  # 类间减少相似度 intra_loss
        sim_max_loss = SimMaxLossv2(num_queries=self.num_queries,
                                    cls_weight=self.cls_branches[-1].weight)  # 类内增大相似度 inter_loss
        class_num = self.num_classes  # todo 加1代表的是加一个背景类别

        # # todo 计算单一的layer的loss
        # queries = hidden_states[1, :, -30:, :]
        # inter_loss = sim_min_loss(queries, class_num)
        # intra_loss = sim_max_loss(queries, class_num)
        # losses['inter_loss'] = inter_loss
        # losses['intra_loss'] = intra_loss

        # todo 计算指定layers的loss均值 效率很低32s50个iter
        # indices_list = [0, 1, 2, 3, 4, 5]
        # inter_loss_total = 0.0
        # intra_loss_total = 0.0
        # for i in indices_list:
        #     queries = hidden_states[i, :, -30:, :]
        #     inter_loss = sim_min_loss(queries, class_num)
        #     intra_loss = sim_max_loss(queries, class_num)
        #     inter_loss_total += inter_loss
        #     intra_loss_total += intra_loss
        #
        # # 计算平均损失
        # losses['inter_loss'] = inter_loss_total / len(indices_list)
        # losses['intra_loss'] = intra_loss_total / len(indices_list)

        # todo 高效计算layers的loss均值
        indices_list = [0]  # base 0
        # indices_list = [0,1,2,3,4,5]
        # 假设 hidden_states 的形状为 (L, B, M, C)
        # indices_list 是一个包含要计算的 L 索引的列表

        # 将所有需要的 queries 组合成 (L*B, M, C) 形状的张量
        queries_batch = torch.cat([hidden_states[i, :, -self.num_queries:, :] for i in indices_list], dim=0)
        # shape of queries_batch = (len(indices_list) * B, 30, C)

        # 现在假设 sim_min_loss 和 sim_max_loss 可以接受 (L*B, M, C) 形状的张量
        # 并返回一个包含批量损失的张量
        inter_loss = sim_min_loss(queries_batch, class_num) * 0.5
        intra_loss = sim_max_loss(queries_batch, class_num) * 1

        # 计算平均损失
        losses['inter_loss'] = inter_loss.sum() / len(indices_list)  # 假设返回的是 (L*B,) 形状的张量
        losses['intra_loss'] = intra_loss.sum() / len(indices_list)  # 假设返回的是 (L*B,) 形状的张量
        return losses

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # extract denoising and matching part of outputs
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds, dn_meta)
        # TODO 这里必须重新写一个方法，不然默认使用匈牙利匹配，大不了直接仿照loss_dn写一个loss_match
        # TODO 这里改成self.loss_match_H,输入暂时不考虑ignore
        loss_dict = self.loss_match(
            all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
            batch_gt_instances, batch_img_metas)  # 这里是使用detr中的loss计算decoder的match目标的loss，内部使用标签分配，TODO：是不是使用匈牙利匹配呢？是的话感觉会很麻烦没法弄固定类别需要改
        # NOTE DETRHead.loss_by_feat but not DeformableDETRHead.loss_by_feat # 这里对decoder多层输出都使用匈牙利匹配了？至少最后一层用了，并且未匹配的queries的结果都是no object
        # is called, because the encoder loss calculations are different
        # between DINO and DeformableDETR.

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)  # 这里对encoder输出的预测进行loss计算，同样使用匈牙利匹配
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
                                  dn_losses_iou[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
        return loss_dict

    def loss_match(self, all_layers_match_cls_scores: Tensor,
                all_layers_match_bbox_preds: Tensor,
                batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                # dn_meta: Dict[str, int]
                   ) -> Tuple[List[Tensor]]:
        """Calculate denoising loss.

        Args:
            # all_layers_denoising_cls_scores (Tensor): Classification scores of
            #     all decoder layers in denoising part, has shape (
            #     num_decoder_layers, bs, num_denoising_queries,
            #     cls_out_channels).
            # all_layers_denoising_bbox_preds (Tensor): Regression outputs of all
            #     decoder layers in denoising part. Each is a 4D-tensor with
            #     normalized coordinate format (cx, cy, w, h) and has shape
            #     (num_decoder_layers, bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            # dn_meta (Dict[str, int]): The dictionary saves information about
            #   group collation, including 'num_denoising_queries' and
            #   'num_denoising_groups'. It will be used for split outputs of
            #   denoising and matching parts and loss calculation.

        Returns:
            Tuple[List[Tensor]]: The loss_dn_cls, loss_dn_bbox, and loss_dn_iou
            of each decoder layers.
        """
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self._loss_match_single,
            all_layers_match_cls_scores,
            all_layers_match_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            # dn_meta=dn_meta
        )

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in \
                zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict
    #
    def _loss_match_single(self, match_cls_scores: Tensor, match_bbox_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        # dn_meta: Dict[str, int]
                           ) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            # dn_cls_scores (Tensor): Classification scores of a single decoder
            #     layer in denoising part, has shape (bs, num_denoising_queries,
            #     cls_out_channels).
            # dn_bbox_preds (Tensor): Regression outputs of a single decoder
            #     layer in denoising part. Each is a 4D-tensor with normalized
            #     coordinate format (cx, cy, w, h) and has shape
            #     (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            # dn_meta (Dict[str, int]): The dictionary saves information about
            #   group collation, including 'num_denoising_queries' and
            #   'num_denoising_groups'. It will be used for split outputs of
            #   denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        # TODO 这个是分配的核心，只要把这个地方改好了就行了，首先查看本来的结果shape，num_total_pos需要注意一下后续要不要改
        cls_reg_targets = self.get_match_targets(batch_gt_instances,
                                              batch_img_metas,match_cls_scores,match_bbox_preds)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets# bs为2时，两张图片目标多的那个图片的正负样本一样多都是100，而gt少的图片只有num_gt*num_group的正样本，剩下都是负样本
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = match_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor) #  TODO 验证一下背景类15是不是变成了十五个0，然后做二分类，做sigmoid而不是softmax 答：从cuda内核来背景类相当于十五个0
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, match_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = match_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou
    #
    def get_match_targets(self, batch_gt_instances: InstanceList,
                       batch_img_metas: dict,match_cls_scores,match_bbox_preds
                          # dn_meta: Dict[str,int]
                          ) -> tuple:
        """Get targets in denoising part for a batch of images.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            # dn_meta (Dict[str, int]): The dictionary saves information about
            #   group collation, including 'num_denoising_queries' and
            #   'num_denoising_groups'. It will be used for split outputs of
            #   denoising and matching parts and loss calculation.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
             self._get_match_targets_single,
             batch_gt_instances,
             batch_img_metas,
            match_cls_scores,
            match_bbox_preds
            # ,
            #  dn_meta=dn_meta
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)
    #
    def _get_match_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict,
                                  match_cls_scores,match_bbox_preds
                                  # dn_meta: Dict[str,int]
                                  ) -> tuple:
        """Get targets in denoising part for one image.

        Args:
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # gt_bboxes = gt_instances.bboxes
        # gt_labels = gt_instances.labels
        # TODO 按组进行匈牙利匹配后在进行结合
        img_h, img_w = img_meta['img_shape']
        factor = match_bbox_preds.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = match_bbox_preds.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(match_bbox_preds)
        bbox_pred = bbox_pred * factor

        pred_instances = InstanceData(scores=match_cls_scores, bboxes=bbox_pred)
        # assigner and sampler TODO 这里使用phind代码进行结果拼接，注意gt_inds拼接结果是否正确，num_gts结果，
        # assign_result = self.assigner.assign(
        #     pred_instances=pred_instances,
        #     gt_instances=gt_instances,
        #     img_meta=img_meta)
        grouped_preds = []
        num_preds_per_group = len(pred_instances) // self.num_classes
        for i in range(self.num_classes):
            start = i * num_preds_per_group
            end = (i + 1) * num_preds_per_group if i < 14 else len(pred_instances)
            grouped_preds.append(pred_instances[start:end])
        assign_results = []
        for i in range(self.num_classes):
            pred_instances_group = grouped_preds[i]
            # Get ground truth instances for this category
            gt_instances_group = gt_instances[gt_instances.labels == i]

            # Perform the assignment operation
            assign_result_t = self.assigner.assign(pred_instances_group, gt_instances_group, img_meta)

            # Correct the assigned_gt_inds
            # if (gt_instances.labels == i).any():
            #     start_index = (gt_instances.labels == i).nonzero()[0]
            #     assign_result.gt_inds[assign_result.gt_inds > 0] += start_index
            if (gt_instances.labels == i).any():
                indices = (gt_instances.labels == i).nonzero(as_tuple=True)[0]
                for j in range(len(indices)):
                    assign_result_t.gt_inds[assign_result_t.gt_inds == (j + 1)] = indices[j] + 1 # TODO 检查逻辑认为就算时反着预测序号，因为是直接按照新序号结果进行修改成原版序号的，所以它的结果也是正确的
            assign_results.append(assign_result_t) # TODO 这里的append是个list，要改，而且，num要求和

        gt_inds = torch.cat([res.gt_inds for res in assign_results], dim=0)
        # max_overlaps = torch.cat([res.max_overlaps for res in assign_results], dim=0)
        labels = torch.cat([res.labels for res in assign_results], dim=0)

        assign_result = AssignResult(
            num_gts=sum(res.num_gts for res in assign_results),
            gt_inds=gt_inds,
            max_overlaps=None,
            labels=labels
        )
        # TODO 添加IS指标，将每轮，每张图，每个decoder层，每个query中的标签分配结果存储在json文件中
        # # 示例用法
        # TODO 存储数据,head6中只有decoder进行6次hungarian matching，虽然encoder也会进行第7此，但是不经过这里
        # 创建文件夹
        # save_path = r'D:\Projects\DINO_mmdet3\mmdetection\tools\IS\test'
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # store_data(img_meta, assign_result, layer_count=6,save_path=save_path)

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        # TODO 这里得到的结果还没转成one hot，我们后续需要进行转化，并且需要得到相应的iou进行计算。
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor

        # TODO 计算 iou 首选获取 xyxy形式的gt和pred，并且都需要会归一化到0-1
        # pos_gt_bboxes_normalized 作为真值结果已经满足条件了。还需要预测结果
        gt_boxes_normalized = gt_bboxes / factor
        bbox_pred_normalized = bbox_cxcywh_to_xyxy(match_bbox_preds)#xyxy形式的预测结果
        pairwise_ious = bbox_overlaps(bbox_pred_normalized, gt_boxes_normalized,is_aligned=False) #需要输入xyxy形式，返回900个iou
        pairwise_ious_pos = pairwise_ious[pos_inds,pos_assigned_gt_inds]# TODO 这里再核对一下pos_inds的意义，后面的pos_assigned_gt_inds感觉不应该加进来？？？？？
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes+1)[:,:-1].to(torch.float32)  # 原本15类别被我巧妙地转化成全0向量
        # labels_onehot[pos_inds] = labels_onehot[pos_inds]*pairwise_ious_pos
        labels_onehot[pos_inds] = labels_onehot[pos_inds]*pairwise_ious_pos.unsqueeze(-1).repeat(1,self.num_classes)



        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        # # TODO 混合精度类型对齐
        # pos_gt_bboxes_targets = pos_gt_bboxes_targets.to(dtype=bbox_targets.dtype)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        # return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
        #         neg_inds)
        return (labels_onehot, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                      all_layers_bbox_preds: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Split outputs of the denoising part and the matching part.

        For the total outputs of `num_queries_total` length, the former
        `num_denoising_queries` outputs are from denoising queries, and
        the rest `num_matching_queries` ones are from matching queries,
        where `num_queries_total` is the sum of `num_denoising_queries` and
        `num_matching_queries`.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'.

        Returns:
            Tuple[Tensor]: a tuple containing the following outputs.

            - all_layers_matching_cls_scores (Tensor): Classification scores
              of all decoder layers in matching part, has shape
              (num_decoder_layers, bs, num_matching_queries, cls_out_channels).
            - all_layers_matching_bbox_preds (Tensor): Regression outputs of
              all decoder layers in matching part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_matching_queries, 4).
            - all_layers_denoising_cls_scores (Tensor): Classification scores
              of all decoder layers in denoising part, has shape
              (num_decoder_layers, bs, num_denoising_queries,
              cls_out_channels).
            - all_layers_denoising_bbox_preds (Tensor): Regression outputs of
              all decoder layers in denoising part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_denoising_queries, 4).
        """
        num_denoising_queries = dn_meta['num_denoising_queries']
        if dn_meta is not None:
            all_layers_denoising_cls_scores = \
                all_layers_cls_scores[:, :, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_matching_cls_scores = \
                all_layers_cls_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                all_layers_bbox_preds[:, :, num_denoising_queries:, :]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds)

    # def assign_multi_group(self,
    #            pred_instances: InstanceData,
    #            gt_instances: InstanceData,
    #            img_meta: Optional[dict] = None,
    #            **kwargs) -> AssignResult:
    #     """Computes one-to-one matching based on the weighted costs.
    #
    #     This method assign each query prediction to a ground truth or
    #     background. The `assigned_gt_inds` with -1 means don't care,
    #     0 means negative sample, and positive number is the index (1-based)
    #     of assigned gt.
    #     The assignment is done in the following steps, the order matters.
    #
    #     1. assign every prediction to -1
    #     2. compute the weighted costs
    #     3. do Hungarian matching on CPU based on the costs
    #     4. assign all to 0 (background) first, then for each matched pair
    #        between predictions and gts, treat this prediction as foreground
    #        and assign the corresponding gt index (plus 1) to it.
    #
    #     Args:
    #         pred_instances (:obj:`InstanceData`): Instances of model
    #             predictions. It includes ``priors``, and the priors can
    #             be anchors or points, or the bboxes predicted by the
    #             previous stage, has shape (n, 4). The bboxes predicted by
    #             the current model or stage will be named ``bboxes``,
    #             ``labels``, and ``scores``, the same as the ``InstanceData``
    #             in other places. It may includes ``masks``, with shape
    #             (n, h, w) or (n, l).
    #         gt_instances (:obj:`InstanceData`): Ground truth of instance
    #             annotations. It usually includes ``bboxes``, with shape (k, 4),
    #             ``labels``, with shape (k, ) and ``masks``, with shape
    #             (k, h, w) or (k, l).
    #         img_meta (dict): Image information.
    #
    #     Returns:
    #         :obj:`AssignResult`: The assigned result.
    #     """
    #     assert isinstance(gt_instances.labels, Tensor)
    #     num_gts, num_preds = len(gt_instances), len(pred_instances)
    #     gt_labels = gt_instances.labels
    #     device = gt_labels.device
    #
    #     # 1. assign -1 by default
    #     assigned_gt_inds = torch.full((num_preds, ),
    #                                   -1,
    #                                   dtype=torch.long,
    #                                   device=device)
    #     assigned_labels = torch.full((num_preds, ),
    #                                  -1,
    #                                  dtype=torch.long,
    #                                  device=device)
    #
    #     if num_gts == 0 or num_preds == 0:
    #         # No ground truth or boxes, return empty assignment
    #         if num_gts == 0:
    #             # No ground truth, assign all to background
    #             assigned_gt_inds[:] = 0
    #         return AssignResult(
    #             num_gts=num_gts,
    #             gt_inds=assigned_gt_inds,
    #             max_overlaps=None,
    #             labels=assigned_labels)
    #
    #     # 2. compute weighted cost
    #     cost_list = []
    #     for match_cost in self.match_costs:
    #         cost = match_cost(
    #             pred_instances=pred_instances,
    #             gt_instances=gt_instances,
    #             img_meta=img_meta)
    #         cost_list.append(cost)
    #     cost = torch.stack(cost_list).sum(dim=0)
    #
    #     # 3. do Hungarian matching on CPU using linear_sum_assignment
    #     cost = cost.detach().cpu()
    #     if linear_sum_assignment is None:
    #         raise ImportError('Please run "pip install scipy" '
    #                           'to install scipy first.')
    #     # TODO 看看这里面有没有nan或inf
    #     if torch.isnan(cost).any() or torch.isinf(cost).any():
    #         print("Tensor contains NaN or Inf!")
    #     matched_row_inds, matched_col_inds = linear_sum_assignment(cost)  #输出值分别是预测/行索引和真值/列索引 An array of row indices and one of corresponding column indices giving the optimal assignment. The cost of the assignment can be computed as ``cost_matrix[row_ind, col_ind].sum()``.
    #     matched_row_inds = torch.from_numpy(matched_row_inds).to(device)  # 这个索引输出出来是最有匹配，sum（loss）最小的一对一匹配方法
    #     matched_col_inds = torch.from_numpy(matched_col_inds).to(device)
    #
    #     # 4. assign backgrounds and foregrounds
    #     # assign all indices to backgrounds first
    #     assigned_gt_inds[:] = 0
    #     # assign foregrounds based on matching results
    #     assigned_gt_inds[matched_row_inds] = matched_col_inds + 1 # 对assinged gt序列加1是因为这里面的内容0为背景，TODO 后续不知到应该怎么立即用，所有真值顺序加一
    #     assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
    #     return AssignResult(
    #         num_gts=num_gts,
    #         gt_inds=assigned_gt_inds, # 看起来是返回的这900的预测结果中每个结果对应的n个真值，是哪个的索引，并且是默认0是不索引，
    #         max_overlaps=None,
    #         labels=assigned_labels)