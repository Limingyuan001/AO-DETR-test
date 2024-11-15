# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector


@MODELS.register_module()
class DetectionTransformer(BaseDetector, metaclass=ABCMeta):
    r"""Base class for Detection Transformer.

    In Detection Transformer, an encoder is used to process output features of
    neck, then several queries interact with the encoder features using a
    decoder and do the regression and classification with the bounding box
    head.

    Args:
        backbone (:obj:`ConfigDict` or dict): Config of the backbone.
        neck (:obj:`ConfigDict` or dict, optional): Config of the neck.
            Defaults to None.
        encoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer encoder. Defaults to None.
        decoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer decoder. Defaults to None.
        bbox_head (:obj:`ConfigDict` or dict, optional): Config for the
            bounding box head module. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict, optional): Config
            of the positional encoding module. Defaults to None.
        num_queries (int, optional): Number of decoder query in Transformer.
            Defaults to 100.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            the bounding box head module. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            the bounding box head module. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 num_queries: int = 100,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None
                 , cam=False  # TODO 添加一个是否可视化的判断
                 , visualization_sampling_point = False # TODO 添加一个是否可视化采样点的判断
                 ) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.cam = cam
        self.visualization_sampling_point = visualization_sampling_point
        # process args
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.encoder = encoder
        self.decoder = decoder
        self.positional_encoding = positional_encoding
        self.num_queries = num_queries

        # init model layers
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(bbox_head)
        self._init_layers()

    @abstractmethod
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        pass

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        img_feats = self.extract_feat(batch_inputs)

        # TODO 判断是否要进行cam可视化
        if self.cam == True:
            import torch
            import matplotlib.pyplot as plt
            from PIL import Image
            import os

            # 假设您的特征层数据存储在变量 img_feats 中
            # img_feats[0] 的形状为 torch.Size([1, 256, 21, 40])

            # 假设 batch_data_samples 是一个包含样本信息的列表
            # 我们取第一个样本的原始尺寸和图像路径
            ori_shape = batch_data_samples[0].ori_shape  # (448, 860)
            img_path = batch_data_samples[0].img_path
            imagename=os.path.basename(img_path)
            # 创建保存图像的文件夹
            save_folder = r"D:\Projects\DINO_mmdet3\mmdetection\tools\cam/"  # 替换为您的文件夹路径
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            # 保存原始图片
            original_img = Image.open(img_path)
            original_img.save(os.path.join(save_folder, imagename))

            # 处理特征层并保存热力图
            feature_layer = img_feats[0]  # 取出特征层

            for i in range(feature_layer.shape[1]):
                # 提取单个特征图
                feature_map = feature_layer[0, i, :, :]

                # 双线性插值调整特征图大小
                feature_map = torch.nn.functional.interpolate(feature_map.unsqueeze(0).unsqueeze(0),
                                                              size=ori_shape,
                                                              mode='bilinear',
                                                              align_corners=False)
                feature_map = feature_map.squeeze(0).squeeze(0)

                # 归一化特征图
                feature_min = feature_map.min()
                feature_max = feature_map.max()
                feature_map = (feature_map - feature_min) / (feature_max - feature_min)

                # 将特征图转换为numpy数组以便可视化
                feature_map_np = feature_map.cpu().numpy()

                # 创建热力图
                plt.imshow(original_img)
                plt.imshow(feature_map_np, cmap='jet', alpha=0.5)  # alpha 控制透明度
                # plt.imshow(feature_map_np, cmap='viridis', alpha=0.5)  # alpha 控制透明度
                # plt.imshow(feature_map_np, cmap='plasma', alpha=0.5)  # alpha 控制透明度

                plt.axis('off')

                # 保存热力图
                plt.savefig(os.path.join(save_folder, f"{imagename}_{i}.png"), bbox_inches='tight', pad_inches=0)
                plt.close()
            feature_map_all_channel = torch.sum(feature_layer,dim=1).unsqueeze(1)
            # 双线性插值调整特征图大小
            feature_map_all_channel = torch.nn.functional.interpolate(feature_map_all_channel,
                                                          size=ori_shape,
                                                          mode='bilinear',
                                                          align_corners=False)
            feature_map_all_channel = feature_map_all_channel.squeeze(0).squeeze(0)

            # 归一化特征图
            feature_min = feature_map_all_channel.min()
            feature_max = feature_map_all_channel.max()
            feature_map_all_channel = (feature_map_all_channel - feature_min) / (feature_max - feature_min)

            # 将特征图转换为numpy数组以便可视化
            feature_map_all_channel_np = feature_map_all_channel.cpu().numpy()

            # 创建热力图
            plt.imshow(original_img)
            plt.imshow(feature_map_all_channel_np, cmap='jet', alpha=0.5)  # alpha 控制透明度
            # plt.imshow(feature_map_all_channel_np, cmap='viridis', alpha=0.5)  # alpha 控制透明度
            # plt.imshow(feature_map_all_channel_np, cmap='plasma', alpha=0.5)  # alpha 控制透明度

            plt.axis('off')

            # 保存热力图
            plt.savefig(os.path.join(save_folder, f"{imagename}_all.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results

    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None) -> Dict:
        """Forward process of Transformer, which includes four steps:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'. We
        summarized the parameters flow of the existing DETR-like detector,
        which can be illustrated as follow:

        .. code:: text

                 img_feats & batch_data_samples
                               |
                               V
                      +-----------------+
                      | pre_transformer |
                      +-----------------+
                          |          |
                          |          V
                          |    +-----------------+
                          |    | forward_encoder |
                          |    +-----------------+
                          |             |
                          |             V
                          |     +---------------+
                          |     |  pre_decoder  |
                          |     +---------------+
                          |         |       |
                          V         V       |
                      +-----------------+   |
                      | forward_decoder |   |
                      +-----------------+   |
                                |           |
                                V           V
                               head_inputs_dict

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                    feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        # todo 4/16 2024为了将deform可能进行可视化进行了接口输入调整调整
        # encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)
        # TODO 给encoder额外加入batch_data_samples参数,需要考虑此函数是否保留了**kwargs接口，
        #  因此将deformable_detr.py中的def forward_encoder加入**kwargs接口,
        #  此外为了好控制是否进行可视化采样点，加入可视化接口visualization，encoder和decoder都需要加一个
        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict,
                                                    batch_data_samples=batch_data_samples,
                                                    visualization=self.visualization_sampling_point)
        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)

        # decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        # TODO 同理给decoder额外加入batch_data_samples参数,需要考虑此函数是否保留了**kwargs接口，
        #  因此将dinov2_2.py中的def forward_decoder加入**kwargs接口
        #  此外为了好控制是否进行可视化采样点，加入可视化接口visualization，encoder和decoder都需要加一个
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict,
                                                    batch_data_samples=batch_data_samples,
                                                    visualization=self.visualization_sampling_point)  # decoder输出包含6层decoder的输出feat，和reference points7个代表是预测结果，但是LFT还需要叠加一次下一层预测偏移量所以这里需要保存下来备用

        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    @abstractmethod
    def pre_transformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """Process image features before feeding them to the transformer.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of encoder
            and the second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              'feat_pos', and other algorithm-specific arguments.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask', and
              other algorithm-specific arguments.
        """
        pass

    @abstractmethod
    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, **kwargs) -> Dict:
        """Forward with Transformer encoder.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output and other algorithm-specific
            arguments.
        """
        pass

    @abstractmethod
    def pre_decoder(self, memory: Tensor, **kwargs) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of decoder
            and the second dict contains the inputs of the bbox_head function.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory', and other algorithm-specific arguments.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which is usually empty, or includes
              `enc_outputs_class` and `enc_outputs_class` when the detector
              support 'two stage' or 'query selection' strategies.
        """
        pass

    @abstractmethod
    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        **kwargs) -> Dict:
        """Forward with Transformer decoder.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (bs, num_queries, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output, `references` including
            the initial and intermediate reference_points, and other
            algorithm-specific arguments.
        """
        pass
