# dataset settings

dataset_type = 'CocoDataset'
data_root = 'D:\Projects\data\PIXray_coco/'  # 'D:\Projects\data\PIXray_coco/'
# TODO mmdetection 3.0后config无法影响数据集类别内容了，直接在D:\Projects\DINO_mmdet3\mmdetection\mmdet\datasets\voc.py中改
# metainfo = {
#     'classes': ('Gun', 'Knife', 'Lighter', 'Battery', 'Pliers', 'Scissors', 'Wrench', 'Hammer', 'Screwdriver', 'Dart', 'Bat', 'Fireworks', 'Saw_blade',
#            'Razor_blade', 'Pressure_vessel'),
#     'palette': [
#         (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
#         (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
#         (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
#         (175, 116, 175), (250, 0, 30), (165, 42, 42)
#     ]
# }
# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    # dict(type='Resize', scale=(320, 320), keep_ratio=False),# TODO 为了测试flops才取消keep ratio
    # dict(type='Resize', scale=(1330, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    # dict(type='Resize', scale=(320, 320), keep_ratio=False), # TODO 为了测试flops才取消keep ratio
    # dict(type='Resize', scale=(1330, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
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
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/pixray_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
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
        ann_file='annotations/pixray_test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader
# test_dataloader = dict(# TODO 本来想直接test和val呢感觉这里啥用了，本来后面就有注释推理的结果可能需要单独进行map计算啥的
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations/instances_train2017.json',
#         data_prefix=dict(img='train2017/'),
#         test_mode=True,
#         pipeline=test_pipeline,
#         backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/pixray_test.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator
# test_evaluator = dict( # TODO 本来想直接test和val呢感觉这里啥用了，本来后面就有注释推理的结果可能需要单独进行map计算啥的
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/instances_train2017.json',
#     metric='bbox',
#     format_only=False,
#     backend_args=backend_args)

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')
