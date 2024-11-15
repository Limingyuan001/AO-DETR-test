_base_ = './DINO_r50_opixray.py.py'

fp16 = dict(loss_scale=512.)
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
pretrained = 'E:\D2E\Projects\DINO_mmdet3\pretrained_model/swin_large_patch4_window12_384_22k.pth'  # noqa

num_levels = 5
model = dict(
    type='DINO',
    num_feature_levels=num_levels,
    num_queries=10,  # num_matching_queries 900

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
        type='DINOHead',
        num_classes=5,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        # loss_cls=dict(
        #     type='QualityFocalLoss',
        #     use_sigmoid=True,
        #     beta=2.0,
        #     loss_weight=1.0),
    ) # 改成了15类
)
# load_from = 'E:\D2E\Projects\DINO_mmdet3\pretrained_model\dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth'
# load_from = "../../configs/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth"
# TODO 这里路径索引不明白了，不知道为啥不对研究一下