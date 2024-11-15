_base_ = './AO-DETR_r50_opixray.py'

fp16 = dict(loss_scale=512.)
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
# pretrained = 'E:\D2E\Projects\DINO_mmdet3\pretrained_model/swin_large_patch4_window12_384_22k.pth'  # noqa

num_levels = 5
model = dict(
    type='DINOv2',
    num_feature_levels=num_levels,
    num_queries=100,  # num_matching_queries 900

    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True, # TODO 不知道干啥的，但是在测试flops的时候需要注释掉
        convert_weights=True,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(in_channels=[192, 384, 768, 1536], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels)))
,
    bbox_head=dict(
        type='DINOHeadv2',
        num_classes=5,
        # loss_cls=dict(
        #     type='FocalLoss',
        #     use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=1.0),  # 2.0 in DeformDETR
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
    ) ,# 改成了15类
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

# load_from = 'E:\D2E\Projects\DINO_mmdet3\pretrained_model\dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth'
# load_from = 'E:\D2E\Projects\DINO_mmdet3\checkpoint/opixray-baseline/AO-DETR-swin-l-q100_fine-tune_640/epoch_12.pth'
load_from = r'E:\D2E\Projects\DINO_mmdet3\checkpoint\opixray-baseline\AO-DETR-swin-l-q100_fine-tune_640_e24\best_pascal_voc_mAP_epoch_3.pth'

# load_from = "../../configs/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth"
# TODO 这里路径索引不明白了，不知道为啥不对研究一下