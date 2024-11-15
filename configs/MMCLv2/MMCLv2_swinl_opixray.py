_base_ = './MMCLv2-4scale_r50_8xb2-12e_opixray.py'

fp16 = dict(loss_scale=512.)
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
pretrained = 'E:\D2E\Projects\DINO_mmdet3\pretrained_model/swin_large_patch4_window12_384_22k.pth'  # noqa

num_levels = 5
num_queries = 10
model = dict(
    type='DINO',
    num_feature_levels=num_levels,
    num_queries=num_queries,  # num_matching_queries 900

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
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(in_channels=[192, 384, 768, 1536], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels)))
,
    bbox_head=dict(
        type='DINOHeadv4',  # done v4 是 MMCLv2的版本
        num_classes=5,  # 改成了15类
        sync_cls_avg_factor=True,
        num_queries=num_queries,  # done mmclv2 swin版本也得再head中加入queries数量
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),

)
 # TODO 保留最佳模型和最新的模型
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        max_keep_ckpts=1,
        save_best='auto'))
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa
#todo 继续训练3个epoch用1e-5
# max_epochs = 4
# train_cfg = dict(
#     type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
#
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')
#
# param_scheduler = [
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[3],
#         gamma=0.1)
# ]
# default_hooks = dict(
#     checkpoint=dict(
#         type='CheckpointHook',
#         max_keep_ckpts=4,
#         save_best='auto'))
# # optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW',
#         lr=0.00001,  # 0.0002 for DeformDETR #todo 继续训练两个epoch用1e-5
#         weight_decay=0.0001),
#     clip_grad=dict(max_norm=0.1, norm_type=2),
#     paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
# )  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa
# load_from = 'E:\D2E\Projects\DINO_mmdet3\pretrained_model\dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth'
# load_from = "../../configs/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth"
# load_from=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\dino\baseline\AO-DETR-swin-l\best_coco_bbox_mAP_epoch_12.pth'
# TODO 这里路径索引不明白了，不知道为啥不对研究一下