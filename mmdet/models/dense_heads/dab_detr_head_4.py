# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch.nn as nn
from mmcv.cnn import Linear
from mmengine.model import bias_init_with_prob, constant_init
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
from ..layers import MLP, inverse_sigmoid
from .conditional_detr_head import ConditionalDETRHead

import torch
import torch.nn as nn
import torch.nn.functional as F

# done 由于head3中intra loss基本上不收敛 因此在head3的基础上进一步对simmax类别内部的对角线掩码

# 提供的 cos_sim 函数
def cos_sim(embedded):
    embedded = F.normalize(embedded, dim=2)
    sim = torch.matmul(embedded, embedded.transpose(1, 2))
    return torch.clamp(sim, min=0.0005, max=0.9995)


# SimMinLoss 类
class SimMinLoss(nn.Module):
    def __init__(self, margin=0.15, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.m = margin
        self.metric = metric
        self.reduction = reduction

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
            sim = cos_sim(embedded)  # sim 的维度是 (B, M, M)

            # 创建掩码并应用到相似度矩阵
            mask = self.create_mask(M, class_num).to(embedded.device)
            mask = mask.unsqueeze(0).expand(B, M, M)  # 扩展掩码到批次大小
            sim = sim.masked_fill(~mask, 0)  # 应用掩码

            loss = -torch.log(1 - sim)
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

# todo 掩码版simmax
class SimMaxLoss(nn.Module):
    def __init__(self, metric='cos', alpha=2.5, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction
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
            # 使用之前定义的 cos_sim 函数计算相似度
            sim = cos_sim(embedded)

            # 应用一个掩码将对角线上的方块的相似度进行保留，其他位置置零，这样能将继续计算类内相似度排名
            mask = self.create_mask(M, class_num).to(embedded.device)
            mask = mask.unsqueeze(0).expand(B, M, M)  # 扩展掩码到批次大小
            sim = sim.masked_fill(mask, 0.00001)  # 应用掩码
            loss = -torch.log(sim)


            # 计算排名权重
            _, indices = sim.sort(descending=True, dim=2)
            _, rank = indices.sort(dim=2)
            rank_weights = torch.exp(-rank.float() * self.alpha)

            # todo 试试要不要对rank进行mask, done 发现没有区别
            rank_weights=rank_weights.masked_fill_(mask, 0)
            # loss1=loss*rank_weights1
            # loss1 = loss1[loss1 > 0]
            # 应用排名权重
            loss = loss * rank_weights

            # 过滤掉对角线上及负损失值
            loss = loss[loss > 0.01]
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
@MODELS.register_module()
class DABDETRHeadv3(ConditionalDETRHead):
    """Head of DAB-DETR. DAB-DETR: Dynamic Anchor Boxes are Better Queries for
    DETR.

    More details can be found in the `paper
    <https://arxiv.org/abs/2201.12329>`_ .
    """

    def _init_layers(self) -> None:
        """Initialize layers of the transformer head."""
        # cls branch
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        # reg branch
        self.fc_reg = MLP(self.embed_dims, self.embed_dims, 4, 3)

    def init_weights(self) -> None:
        """initialize weights."""
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)
        constant_init(self.fc_reg.layers[-1], 0., bias=0.)

    def forward(self, hidden_states: Tensor,
                references: Tensor) -> Tuple[Tensor, Tensor]:
        """"Forward function.

        Args:
            hidden_states (Tensor): Features from transformer decoder. If
                `return_intermediate_dec` is True output has shape
                (num_decoder_layers, bs, num_queries, dim), else has shape (1,
                bs, num_queries, dim) which only contains the last layer
                outputs.
            references (Tensor): References from transformer decoder. If
                `return_intermediate_dec` is True output has shape
                (num_decoder_layers, bs, num_queries, 2/4), else has shape (1,
                bs, num_queries, 2/4)
                which only contains the last layer reference.
        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - layers_cls_scores (Tensor): Outputs from the classification head,
              shape (num_decoder_layers, bs, num_queries, cls_out_channels).
              Note cls_out_channels should include background.
            - layers_bbox_preds (Tensor): Sigmoid outputs from the regression
              head with normalized coordinate format (cx, cy, w, h), has shape
              (num_decoder_layers, bs, num_queries, 4).
        """
        layers_cls_scores = self.fc_cls(hidden_states)
        references_before_sigmoid = inverse_sigmoid(references, eps=1e-3)
        tmp_reg_preds = self.fc_reg(hidden_states)
        tmp_reg_preds[..., :references_before_sigmoid.
                      size(-1)] += references_before_sigmoid
        layers_bbox_preds = tmp_reg_preds.sigmoid()
        return layers_cls_scores, layers_bbox_preds

    # todo 由于这里没有loss，而是直接使用conditional的loss，为了不引起混乱，我们进行继承后加入聚类排斥损失函数
    def loss(self, hidden_states: Tensor, references: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            hidden_states (Tensor): Features from the transformer decoder, has
                shape (num_decoder_layers, bs, num_queries, dim).
            references (Tensor): References from the transformer decoder, has
               shape (num_decoder_layers, bs, num_queries, 2).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        loss_inputs = outs + (batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        # todo 加入聚类和排斥损失
        sim_min_loss = SimMinLoss()  # 类间减少相似度 intra_loss
        sim_max_loss = SimMaxLoss()  # 类内增大相似度 inter_loss
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
        indices_list = [0]
        # 假设 hidden_states 的形状为 (L, B, M, C)
        # indices_list 是一个包含要计算的 L 索引的列表

        # 将所有需要的 queries 组合成 (L*B, M, C) 形状的张量
        queries_batch = torch.cat([hidden_states[i, :, -300:, :] for i in indices_list], dim=0)
        # shape of queries_batch = (len(indices_list) * B, 30, C)

        # 现在假设 sim_min_loss 和 sim_max_loss 可以接受 (L*B, M, C) 形状的张量
        # 并返回一个包含批量损失的张量
        inter_loss = sim_min_loss(queries_batch, class_num) * 0.5
        intra_loss = sim_max_loss(queries_batch, class_num)

        # 计算平均损失
        losses['inter_loss'] = inter_loss.sum() / len(indices_list)  # 假设返回的是 (L*B,) 形状的张量
        losses['intra_loss'] = intra_loss.sum() / len(indices_list)  # 假设返回的是 (L*B,) 形状的张量

        return losses
    def predict(self,
                hidden_states: Tensor,
                references: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network. Over-write
        because img_metas are needed as inputs for bbox_head.

        Args:
            hidden_states (Tensor): Feature from the transformer decoder, has
                shape (num_decoder_layers, bs, num_queries, dim).
            references (Tensor): references from the transformer decoder, has
                shape (num_decoder_layers, bs, num_queries, 2/4).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        last_layer_hidden_state = hidden_states[-1].unsqueeze(0)
        last_layer_reference = references[-1].unsqueeze(0)
        outs = self(last_layer_hidden_state, last_layer_reference)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions
