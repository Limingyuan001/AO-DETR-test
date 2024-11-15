# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from ..utils import multi_apply
from .deformable_detr_head import DeformableDETRHead
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

            # 应用排名权重
            loss = loss * rank_weights

            # 过滤掉对角线上及负损失值
            # loss = loss[loss > 0.01]  # todo 默认是m=0.01最好，但是实验需要调参 5/5/2024
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
class DINOHeadv3(DeformableDETRHead):
    r"""Head of the DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2203.03605>`_ .
    """

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

        # todo 加入聚类和排斥损失
        sim_min_loss = SimMinLoss()  # 类间减少相似度 intra_loss; IMC loss
        sim_max_loss = SimMaxLoss()  # 类内增大相似度 inter_loss; MIE loss
        class_num = self.num_classes  # todo 如果加1代表的是加一个背景类别


        # todo 高效计算layers的loss均值
        # indices_list 是一个包含要计算的 L 索引的列表, 即论文中的target layer set
        indices_list = [0]
        # 假设 hidden_states 的形状为 (L, B, M, C)


        # 将所有需要的 queries 组合成 (L*B, M, C) 形状的张量
        queries_batch = torch.cat([hidden_states[i, :, -30:, :] for i in indices_list], dim=0)
        # shape of queries_batch = (len(indices_list) * B, 30, C)

        # 现在假设 sim_min_loss 和 sim_max_loss 可以接受 (L*B, M, C) 形状的张量
        # 并返回一个包含批量损失的张量
        inter_loss = sim_min_loss(queries_batch, class_num)*0.5
        intra_loss = sim_max_loss(queries_batch, class_num)*1

        # 计算平均损失, 并添加MMCL loss
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

        loss_dict = super(DeformableDETRHead, self).loss_by_feat(
            all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)  # 这里是使用detr中的loss计算decoder的match目标的loss，内部使用标签分配，TODO：是不是使用匈牙利匹配呢？是的话感觉会很麻烦没法弄固定类别需要改
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

    def loss_dn(self, all_layers_denoising_cls_scores: Tensor,
                all_layers_denoising_bbox_preds: Tensor,
                batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                dn_meta: Dict[str, int]) -> Tuple[List[Tensor]]:
        """Calculate denoising loss.

        Args:
            all_layers_denoising_cls_scores (Tensor): Classification scores of
                all decoder layers in denoising part, has shape (
                num_decoder_layers, bs, num_denoising_queries,
                cls_out_channels).
            all_layers_denoising_bbox_preds (Tensor): Regression outputs of all
                decoder layers in denoising part. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and has shape
                (num_decoder_layers, bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[List[Tensor]]: The loss_dn_cls, loss_dn_bbox, and loss_dn_iou
            of each decoder layers.
        """
        return multi_apply(
            self._loss_dn_single,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta)

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                              batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets# bs为2时，两张图片目标多的那个图片的正负样本一样多都是100，而gt少的图片只有num_gt*num_group的正样本，剩下都是负样本
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # TODO 此处配套head6，由于decoder中的cls不受head6影响，而这里将encoder的cls会因为配置文件同步成QFL，真值需要改成iou太麻烦了， 直接固定成focal loss试试
        # if len(cls_scores) > 0:
        #     loss_cls = self.loss_cls(
        #         cls_scores, labels, label_weights, avg_factor=cls_avg_factor) #  TODO 验证一下背景类15是不是变成了十五个0，然后做二分类，做sigmoid而不是softmax 答：从cuda内核来背景类相当于十五个0
        # else:
        #     loss_cls = torch.zeros(
        #         1, dtype=cls_scores.dtype, device=cls_scores.device)
        if len(cls_scores) > 0:
            loss_cls = self.focal_loss_cls(
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
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def get_dn_targets(self, batch_gt_instances: InstanceList,
                       batch_img_metas: dict, dn_meta: Dict[str,
                                                            int]) -> tuple:
        """Get targets in denoising part for a batch of images.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

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
             self._get_dn_targets_single,
             batch_gt_instances,
             batch_img_metas,
             dn_meta=dn_meta)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_dn_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str,
                                                             int]) -> tuple:
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
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(
                num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = \
                gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2

        # label targets
        labels = gt_bboxes.new_full((num_denoising_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)  # label真值：正样本为正常类别，负样本类别为15
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)#label ：正负样本权重都是,以及padding都是1

        # bbox targets
        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)# box ：负样本和padding的权重为0，正样本为1
        bbox_weights[pos_inds] = 1.0
        img_h, img_w = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)#box真值：负样本为0000，正样本为gt
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
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
