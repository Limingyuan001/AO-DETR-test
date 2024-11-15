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
# TODO head6 在head5的基础上，引入iou作为label的真值，并使用QFL来替换Focal loss。
import json
import os
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
class DINOHeadv2(DINOHead):
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