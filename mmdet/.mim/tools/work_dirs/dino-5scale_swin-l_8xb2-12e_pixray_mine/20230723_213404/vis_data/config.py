dataset_type = 'CocoDataset'
data_root = 'D:\\Projects\\data\\PIXray_coco/'
metainfo = dict(
    classes=('Gun', 'Knife', 'Lighter', 'Battery', 'Pliers', 'Scissors',
             'Wrench', 'Hammer', 'Screwdriver', 'Dart', 'Bat', 'Fireworks',
             'Saw_blade', 'Razor_blade', 'Pressure_vessel'),
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
             (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
             (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
             (175, 116, 175), (250, 0, 30), (165, 42, 42)])
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[{
            'type':
            'RandomChoiceResize',
            'scales': [(192, 320), (205, 320), (218, 320), (230, 320),
                       (243, 320), (256, 320), (269, 320), (282, 320),
                       (294, 320), (307, 320), (320, 320)],
            'keep_ratio':
            True
        }],
                    [{
                        'type': 'RandomChoiceResize',
                        'scales': [(160, 1008), (200, 1008), (240, 1008)],
                        'keep_ratio': True
                    }, {
                        'type': 'RandomCrop',
                        'crop_type': 'absolute_range',
                        'crop_size': (150, 200),
                        'allow_negative_crop': True
                    }, {
                        'type':
                        'RandomChoiceResize',
                        'scales': [(192, 320), (205, 320), (218, 320),
                                   (230, 320), (243, 320), (256, 320),
                                   (269, 320), (282, 320), (294, 320),
                                   (307, 320), (320, 320)],
                        'keep_ratio':
                        True
                    }]]),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
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
    dataset=dict(
        type='CocoDataset',
        data_root='D:\\Projects\\data\\PIXray_coco/',
        ann_file='annotations/pixray_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='RandomChoice',
                transforms=[[{
                    'type':
                    'RandomChoiceResize',
                    'scales': [(192, 320), (205, 320), (218, 320), (230, 320),
                               (243, 320), (256, 320), (269, 320), (282, 320),
                               (294, 320), (307, 320), (320, 320)],
                    'keep_ratio':
                    True
                }],
                            [{
                                'type': 'RandomChoiceResize',
                                'scales': [(160, 1008), (200, 1008),
                                           (240, 1008)],
                                'keep_ratio': True
                            }, {
                                'type': 'RandomCrop',
                                'crop_type': 'absolute_range',
                                'crop_size': (150, 200),
                                'allow_negative_crop': True
                            }, {
                                'type':
                                'RandomChoiceResize',
                                'scales': [(192, 320), (205, 320), (218, 320),
                                           (230, 320), (243, 320), (256, 320),
                                           (269, 320), (282, 320), (294, 320),
                                           (307, 320), (320, 320)],
                                'keep_ratio':
                                True
                            }]]),
            dict(type='PackDetInputs')
        ],
        backend_args=None))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='D:\\Projects\\data\\PIXray_coco/',
        ann_file='annotations/pixray_test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(320, 320), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='D:\\Projects\\data\\PIXray_coco/',
        ann_file='annotations/pixray_test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(320, 320), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='D:\\Projects\\data\\PIXray_coco/annotations/pixray_test.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='D:\\Projects\\data\\PIXray_coco/annotations/pixray_test.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = 'E:\\D2E\\Projects\\DINO_mmdet3\\checkpoint\\dino\\swinL_pixray\\test\\epoch_1.pth'
resume = False
model = dict(
    type='DINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        convert_weights=True),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=5),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=5, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=5, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='DINOHead',
        num_classes=15,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300),
    num_feature_levels=5)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))))
max_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]
auto_scale_lr = dict(base_batch_size=2)
fp16 = dict(loss_scale=512.0)
num_levels = 5
launcher = 'none'
work_dir = './work_dirs\\dino-5scale_swin-l_8xb2-12e_pixray_mine'
