# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmengine.model import bias_init_with_prob, constant_init
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList
from ..layers import inverse_sigmoid
from .detr_head import DETRHead

import torch.nn.functional as F
# done 由于head3中intra loss基本上不收敛 因此在head3的基础上进一步对simmax类别内部的对角线掩码

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

            # todo B：相较于dinov4_head_4.py对于类间损失，放弃样本对之间的相似度sim，直接使用每个样本和类别头原型的余弦相似度计算loss。
            normalized_weight = F.normalize(self.cls_weight, p=2, dim=1)  # shape: (15, 256)

            # 计算相似度矩阵，使用矩阵乘法实现
            similarity_matrix = torch.mm(normalized_weight, normalized_weight.t())  # shape: (15, 15)
            similarity_matrix = torch.clamp(similarity_matrix, min=0.0005, max=0.9995)

            # 使用最近邻插值上采样
            upsampled_similarity_matrix = F.interpolate(similarity_matrix.unsqueeze(0).unsqueeze(0),
                                        size=(self.num_queries, self.num_queries), mode='nearest').squeeze(0).squeeze(0)
            usm = upsampled_similarity_matrix.masked_fill(~mask, 0)  # 应用掩码upsampled_similarity_matrix
            loss = -torch.log(1 - usm)
            # loss = -torch.log(1 - torch.pow(usm,3))
            # todo 还可以使用很多形式

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
            sim = torch.clamp(similarity_matrix, min=0.0005, max=0.9995)

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
@MODELS.register_module()
class DeformableDETRHeadv4(DETRHead):
    r"""Head of DeformDETR: Deformable DETR: Deformable Transformers for
    End-to-End Object Detection.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        share_pred_layer (bool): Whether to share parameters for all the
            prediction layers. Defaults to `False`.
        num_pred_layer (int): The number of the prediction layers.
            Defaults to 6.
        as_two_stage (bool, optional): Whether to generate the proposal
            from the outputs of encoder. Defaults to `False`.
    """

    def __init__(self,
                 *args,
                 share_pred_layer: bool = False,
                 num_pred_layer: int = 6,
                 as_two_stage: bool = False,
                 **kwargs) -> None:
        self.share_pred_layer = share_pred_layer
        self.num_pred_layer = num_pred_layer
        self.as_two_stage = as_two_stage

        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)  # 这里看出每个decoder层都会用特征进行类别预测对应的iterative box
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)#self.reg_branches跟第一次进行偏差tmp预测是同一个线形预测模块
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference  # 这里就是Dino的LFT，因为ref多存了一个开头，后面六个ref和hidden特征是对齐的，所以这里ref是再hidden前一个，正好是用后面一个特征预测偏移并相加
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
                Only when `as_two_stage` is `True` it would be passed in,
                otherwise it would be `None`.
            enc_outputs_coord (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h). Only when `as_two_stage`
                is `True` it would be passed in, otherwise it would be `None`.
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
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
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)

        # todo 类间使用原本mmcl，类内使用新版特征和原型相似度
        sim_min_loss = SimMinLossv2(num_queries=self.num_queries, cls_weight=self.cls_branches[-1].weight)  # 类间减少相似度 intra_loss
        sim_max_loss = SimMaxLossv2(num_queries=self.num_queries, cls_weight=self.cls_branches[-1].weight)  # 类内增大相似度 inter_loss
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
        # queries_batch = torch.cat([hidden_states[i, :, -30:, :] for i in indices_list], dim=0)
        # shape of queries_batch = (len(indices_list) * B, 30, C)
        queries_batch = torch.cat([hidden_states[i, :, -self.num_queries:, :] for i in indices_list], dim=0)


        # 现在假设 sim_min_loss 和 sim_max_loss 可以接受 (L*B, M, C) 形状的张量
        # 并返回一个包含批量损失的张量
        inter_loss = sim_min_loss(queries_batch, class_num) * 0.5
        intra_loss = sim_max_loss(queries_batch, class_num)

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
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
                Only when `as_two_stage` is `True` it would be passes in,
                otherwise, it would be `None`.
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h). Only when `as_two_stage`
                is `True` it would be passed in, otherwise it would be `None`.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_dict = super().loss_by_feat(all_layers_cls_scores,
                                         all_layers_bbox_preds,
                                         batch_gt_instances, batch_img_metas,
                                         batch_gt_instances_ignore)

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            proposal_gt_instances = copy.deepcopy(batch_gt_instances)
            for i in range(len(proposal_gt_instances)):
                proposal_gt_instances[i].labels = torch.zeros_like(
                    proposal_gt_instances[i].labels)
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=proposal_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou
        return loss_dict

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(hidden_states, references)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        batch_img_metas: List[Dict],
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (num_decoder_layers, bs, num_queries,
                4) with the last dimension arranged as (cx, cy, w, h).
            batch_img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Default `False`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]  # 这里的【-1】相当于只取最后一个decoder layers的预测结果

        # TODO 为了可视化不同层的AP,我们更改这里的代码,decoder一共六层,-1=5代表最后一层
        # cls_scores = all_layers_cls_scores[0]
        # bbox_preds = all_layers_bbox_preds[0]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list
