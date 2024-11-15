_base_ = [
    '../_base_/datasets/opixray_detection.py', '../_base_/default_runtime.py'
]
# -------------------------------------------------#
# 记录一下更改内容
# 第一，改上面的数据集
# 第二，将frozen_block=1注释掉
# 第三，RandomChoiceResize中的图片比例不符合320*320，需要更改
# 第四，类别数从80-->15
# 第五，由于只有一张显卡，所以auto_scale_lr的batch_size=2
# 第六，改为加载全部预训练模型load_from
# 后续可能需要更改学习率，num_workers,num_queries,以及多少轮测试与保存等
# -------------------------------------------------#
train_batch_size=2
visualization_sampling_point=False #默认进行采样点可视化
cam=False
model = dict(
    type='DINOv2',
    cam=cam,  # 添加一个参数是否要进行可视化，默认不进行cam可视化
    visualization_sampling_point=visualization_sampling_point,  # TODO decoder是否进行可视化采样点函数
    num_queries=100,  # num_matching_queries 900
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        # frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        # init_cfg=dict(type='Pretrained',
        #               checkpoint=r'E:\D2E\Projects\DINO_mmdet3\pretrained_model\resnet50-0676ba61.pth')

        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        # visualization_sampling_point=visualization_sampling_point,  # TODO encoder是否进行可视化采样点函数
        num_layers=6,  # num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        # visualization_sampling_point=visualization_sampling_point,  # TODO decoder是否进行可视化采样点函数
        num_layers=6,  # num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)
            ,
        # flag_cfg = dict(self_attn_flag=True,
        #         ffn0_flag=True,
        #         decoder_cross_attn_flag=False,
        #         decoder_deform_attn_flag=True
        #         ) # TODO 注意这行存在的话就代表是用ffn0，否则使用self-attn
        ),  # 0.1 for DeformDETR

        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DINOHeadv2',
        num_classes=5, # 改成了5类
        sync_cls_avg_factor=True,
        # loss_cls=dict(
            # type='FocalLoss',
            # use_sigmoid=True,
            # gamma=2.0,
            # alpha=0.25,
            # loss_weight=1.0),  # 2.0 in DeformDETR
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    # train_cfg=dict(
    #     assigner=dict(
    #         type='HungarianAssigner',
    #         match_costs=[
    #             dict(type='FocalLossCost', weight=2.0),
    #             dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
    #             dict(type='IoUCost', iou_mode='giou', weight=2.0)
    #         ])),
    # test_cfg=dict(max_per_img=300))  # 100 for DeformDETR
train_cfg = dict(
    assigner=dict(
        type='HungarianAssigner',
        match_costs=[
            dict(type='FocalLossCost', weight=2.0),
            dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            dict(type='IoUCost', iou_mode='giou', weight=2.0)
        ])),
            test_cfg = dict(max_per_img=300)
)
dataset_type = 'VOCDataset'
data_root = 'D:\Projects\data\OPIXray_voc/'  # 'D:\Projects\data\PIXray_coco/'
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice', # 这里是训练的数据增强，按照下列两种方案随机进行选取每张图片如何增强
        transforms=[
            # 方案一，直接随机resize （keep ratio）
            [
                dict(
                    type='RandomChoiceResize',
                    # scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                    #         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                    #         (736, 1333), (768, 1333), (800, 1333)],
# scales=[(192, 320), (205, 320), (218, 320), (230, 320),
#                             (243, 320), (256, 320), (269, 320), (282, 320),
#                             (294, 320), (307, 320), (320, 320)],
scales = [(384, 640), (410, 640), (436, 640), (460, 640),
                 (486, 640), (512, 640), (538, 640), (564, 640),
                 (588, 640), (614, 640), (640, 640)],
                    keep_ratio=True)
            ],
            # 方案二，先随机resize，然后随机剪裁，最后进行随机resize
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    # scales=[(400, 4200), (500, 4200), (600, 4200)],
                    # scales=[(160, 1008), (200, 1008), (240, 1008)],
scales = [(320, 2016), (400, 2016), (480, 2016)],

                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    # crop_size=(384, 600),
                    # crop_size=(150, 200),  # 320/800*384 , 600/1333*320
                    crop_size=(300, 400),  # 320/800*384 , 600/1333*320

                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    # scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                    #         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                    #         (736, 1333), (768, 1333), (800, 1333)],
# scales=[(192, 320), (205, 320), (218, 320), (230, 320),
#                             (243, 320), (256, 320), (269, 320), (282, 320),
#                             (294, 320), (307, 320), (320, 320)],
scales = [(384, 640), (410, 640), (436, 640), (460, 640),
                 (486, 640), (512, 640), (538, 640), (564, 640),
                 (588, 640), (614, 640), (640, 640)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]
# train_batch_size=2
# train_dataloader = dict(
#     batch_size=train_batch_size,
#     dataset=dict(
#         filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    # dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    # dict(type='Resize', scale=(320, 320), keep_ratio=True),
    # dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),

    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    # dataset=dict(
    #     type='RepeatDataset',
    #     times=3,
    #     dataset=dict(
    #         type='ConcatDataset',
    #         # VOCDataset will add different `dataset_type` in dataset.metainfo,
    #         # which will get error if using ConcatDataset. Adding
    #         # `ignore_keys` can avoid this error.
    #         ignore_keys=['dataset_type'],
    dataset= dict(type=dataset_type,
                    data_root=data_root,
                    ann_file='VOC2007/ImageSets/Main/train.txt',
                    data_prefix=dict(sub_data_root='VOC2007/'),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32),
                    pipeline=train_pipeline,
                    backend_args=backend_args))


val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

# Pascal VOC2007 uses `11points` as default evaluate mode, while PASCAL
# VOC2012 defaults to use 'area'.
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR # todo 正常是0.0001但是为了直接续上最后一轮的0.00001这里改了，并且把预训练模型改成了12轮的
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa
max_epochs = 15
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]
 # TODO 保留最佳模型和最新的模型
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        max_keep_ckpts=4,
        save_best='auto'))
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=train_batch_size)
load_from = 'E:\D2E\Projects\DINO_mmdet3\pretrained_model\dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
