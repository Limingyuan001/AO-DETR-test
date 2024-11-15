_base_ = [
    '../_base_/datasets/pixray_detection.py', '../_base_/default_runtime.py'
]
train_batch_size=2

model = dict(
    type='DABDETR',
    num_queries=30,
    with_random_refpoints=False,
    num_patterns=0,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    # backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(3, ),
    #     # out_indices=(2, ), # todo试试这样会不会好用
    #     # frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=False),
    #     norm_eval=True,
    #     style='pytorch',
    #     # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    #     init_cfg=dict(type='Pretrained',
    #                   checkpoint=r'E:\D2E\Projects\DINO_mmdet3\pretrained_model\resnet50-0676ba61.pth')
    # ),
    # neck=dict(
    #     type='ChannelMapper',
    #     in_channels=[2048],
    #     # in_channels=[1024],
    #     kernel_size=1,
    #     out_channels=256,
    #     act_cfg=None,
    #     norm_cfg=None,
    #     num_outs=1),
    # todo 尝试多尺度融合
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        # frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0., batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.,
                act_cfg=dict(type='PReLU')))),
    decoder=dict(
        num_layers=6,
        query_dim=4,
        query_scale_type='cond_elewise',
        with_modulated_hw_attn=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                attn_drop=0.,
                proj_drop=0.,
                cross_attn=False),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                attn_drop=0.,
                proj_drop=0.,
                cross_attn=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.,
                act_cfg=dict(type='PReLU'))),
        return_intermediate=True),
    positional_encoding=dict(num_feats=128, temperature=20, normalize=True),
    bbox_head=dict(
        type='DABDETRHead',
        num_classes=15,
        embed_dims=256,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2., eps=1e-8),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='RandomChoice',
#         transforms=[[
#             dict(
#                 type='RandomChoiceResize',
#                 scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                         (736, 1333), (768, 1333), (800, 1333)],
#                 keep_ratio=True)
#         ],
#                     [
#                         dict(
#                             type='RandomChoiceResize',
#                             scales=[(400, 1333), (500, 1333), (600, 1333)],
#                             keep_ratio=True),
#                         dict(
#                             type='RandomCrop',
#                             crop_type='absolute_range',
#                             crop_size=(384, 600),
#                             allow_negative_crop=True),
#                         dict(
#                             type='RandomChoiceResize',
#                             scales=[(480, 1333), (512, 1333), (544, 1333),
#                                     (576, 1333), (608, 1333), (640, 1333),
#                                     (672, 1333), (704, 1333), (736, 1333),
#                                     (768, 1333), (800, 1333)],
#                             keep_ratio=True)
#                     ]]),
#     dict(type='PackDetInputs')
# ]
# todo 640训练
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
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    # dict(type='Resize', scale=(800, 800), keep_ratio=True),

    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
dataset_type = 'CocoDataset'
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        # data_root=data_root,
        ann_file='annotations/pixray_test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

# todo 320训练
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='RandomChoice', # 这里是训练的数据增强，按照下列两种方案随机进行选取每张图片如何增强
#         transforms=[
#             # 方案一，直接随机resize （keep ratio）
#             [
#                 dict(
#                     type='RandomChoiceResize',
#                     # scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                     #         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                     #         (736, 1333), (768, 1333), (800, 1333)],
# scales=[(192, 320), (205, 320), (218, 320), (230, 320),
#                             (243, 320), (256, 320), (269, 320), (282, 320),
#                             (294, 320), (307, 320), (320, 320)],
#                     keep_ratio=True)
#             ],
#             # 方案二，先随机resize，然后随机剪裁，最后进行随机resize
#             [
#                 dict(
#                     type='RandomChoiceResize',
#                     # The radio of all image in train dataset < 7
#                     # follow the original implement
#                     # scales=[(400, 4200), (500, 4200), (600, 4200)],
#                     scales=[(160, 1008), (200, 1008), (240, 1008)],
#
#                     keep_ratio=True),
#                 dict(
#                     type='RandomCrop',
#                     crop_type='absolute_range',
#                     # crop_size=(384, 600),
#                     crop_size=(150, 200),  # 320/800*384 , 600/1333*320
#
#                     allow_negative_crop=True),
#                 dict(
#                     type='RandomChoiceResize',
#                     # scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                     #         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                     #         (736, 1333), (768, 1333), (800, 1333)],
# scales=[(192, 320), (205, 320), (218, 320), (230, 320),
#                             (243, 320), (256, 320), (269, 320), (282, 320),
#                             (294, 320), (307, 320), (320, 320)],
#                     keep_ratio=True)
#             ]
#         ]),
#     dict(type='PackDetInputs')
# ]
# train_dataloader = dict(
#     dataset=dict(
#         filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

# learning policy
max_epochs = 200
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=6)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]
 # TODO 保留最佳模型和最新的模型
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        max_keep_ckpts=1,
        save_best='auto'))
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=train_batch_size, enable=False)
