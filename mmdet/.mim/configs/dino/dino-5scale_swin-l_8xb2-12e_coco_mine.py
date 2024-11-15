_base_ = './dino-4scale_r50_8xb2-12e_coco_mine.py'

fp16 = dict(loss_scale=512.)
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
num_levels = 5
model = dict(
    num_feature_levels=num_levels,
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
        with_cp=True,
        convert_weights=True,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(in_channels=[192, 384, 768, 1536], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels))))
load_from = 'E:\D2E\Projects\DINO_mmdet3\pretrained_model\dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth'
# load_from = "../../configs/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth"
# TODO 这里路径索引不明白了，不知道为啥不对研究一下