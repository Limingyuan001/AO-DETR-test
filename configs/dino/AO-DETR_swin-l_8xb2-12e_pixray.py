_base_ = './AO-DETR_r50_pixray.py'
# -------------------------------------------------#
# 记录一下更改内容
# 第一，改上面的数据集
# 第二，将frozen_block=1注释掉
# 第三，RandomChoiceResize中的图片比例不符合320*320，需要更改
# 第四，类别数从80-->15
# 第五，由于只有一张显卡，所以auto_scale_lr的batch_size=2 在上层文件中./pixray_mine，需要和configs/_base_/datasets/pixray_detection.py中的batchsize一起改
# 第六，改为加载全部预训练模型load_from
# 后续可能需要更改学习率，num_workers,num_queries,以及多少轮测试与保存等
# -------------------------------------------------#
fp16 = dict(loss_scale=512.)
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
num_levels = 5
model = dict(
    num_queries=30,  # num_matching_queries 900
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
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(in_channels=[192, 384, 768, 1536], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels))))
# load_from = 'E:\D2E\Projects\DINO_mmdet3\pretrained_model\dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth'

# TODO 这里路径索引不明白了，不知道为啥不对研究一下