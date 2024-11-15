# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    # r50 pixray
    # parser.add_argument('--config',default='../configs/dino/dino-5scale_swin-l_8xb2-12e_pixray_mine_q100.py', help='test config file path')
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\dino\swinL_pixray_q100\02\epoch_10.pth',
    #                     help='checkpoint file')
    # parser.add_argument('--config',default='../configs/dino/dino-4scale_r50_8xb2-12e_coco_mine.py', help='test config file path')
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\dino\r50_coco\01\epoch_1.pth',
    #                     help='checkpoint file')

    # swin-L pixray DINO
    # parser.add_argument('--config', default='../configs/dino/dino-5scale_swin-l_8xb2-12e_pixray_mine_q900.py', help='train config file path')
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\dino\swinL_pixray_q900\basline\epoch_12.pth',
    #                     help='checkpoint file')
    # # swin-L pixray DINOv2
    # parser.add_argument('--config', default='../configs/dino/dinov2-5scale_swin-l_8xb2-12e_pixray_mine_q900.py',
    #                     help='train config file path')
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\dinov2\swinL_pixray_q900\51\epoch_12.pth',
    #                     help='checkpoint file')

    # r50 coco mine2 scale=(1333, 800)
    # parser.add_argument('--config', default='../configs/dino/dino-4scale_r50_8xb2-12e_coco_mine2.py',
    #                     help='train config file path')
    # # parser.add_argument('--checkpoint', default='E:\D2E\Projects\DINO_mmdet3\checkpoint/dino/r50_coco_mine2/baseline/epoch_1.pth',
    # #                     help='the dir to save logs and models')
    # parser.add_argument('--checkpoint', default=r'E:\D2E\Projects\DINO_mmdet3\pretrained_model\dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth',
    #                     help='the dir to save logs and models')
    # r50 pixray backbone
    # parser.add_argument('--config', default='../configs/dino/dino-4scale_r50_8xb2-12e_pixray_mine68.py',
    #                     help='train config file path')
    # parser.add_argument('--checkpoint', default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\dino\r50_pixray\106\best_coco_bbox_mAP_epoch_12.pth',
    #                     help='the dir to save logs and models')
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\dino\r50_pixray\70\epoch_12.pth',
    #                     help='the dir to save logs and models') # 300 query t-sne
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\dino\r50_pixray\69\epoch_12.pth',
    #                     help='the dir to save logs and models') # 300 query t-sne
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\dino\r50_pixray\97\epoch_12.pth',
    #                     help='the dir to save logs and models') # LFT+CWA

    # opixray DINOr50  backbone
    # parser.add_argument('--config', default='D:\Projects\DINO_mmdet3\mmdetection\configs\dino\DINO_r50_opixray.py.py',
    #                     help='train config file path')
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\opixray-baseline\dino-r50\epoch_12.pth',
    #                     help='the dir to save logs and models')
    # opixray DINO swin-l
    # parser.add_argument('--config', default='D:\Projects\DINO_mmdet3\mmdetection\configs\dino\DINO_swin-l_opixray.py',
    #                     help='train config file path')
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint/opixray-baseline/dino-swin-l\epoch_1.pth',
    #                     help='the dir to save logs and models')
    # AO-DETR swin-l opixray fine-tune 640 冲best
    # parser.add_argument('--config', default='D:\Projects\DINO_mmdet3\mmdetection/configs/dino/AO-DETR_swin-l_opixray_fine-tune_640.py', help='train config file path')
    # parser.add_argument('--checkpoint', default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\opixray-baseline\AO-DETR-swin-l-q100_fine-tune_640_e24\best_pascal_voc_mAP_epoch_3.pth',
    #                     help='the dir to save logs and models')

    # AO-DETRv2 可视化t-sne dino r50 pixray q300+loss和不加loss
    # 由于当时实验7保存到了6文件夹中因此只能这样操作
    # parser.add_argument('--config',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\r50_pixray\06\AO-DETRv2_r50_pixray.py', help='test config file path')
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\r50_pixray\06\best_coco_bbox_mAP_epoch_12.pth',
    #                     help='checkpoint file')
    # parser.add_argument('--config',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\r50_pixray\basline300q\AO-DETRv2_r50_pixray.py', help='test config file path')
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\r50_pixray\basline300q\best_coco_bbox_mAP_epoch_12.pth',
    #                     help='checkpoint file')
    # deform
    # parser.add_argument('--config',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\generalization\deform-detr\r50q300+head4\deformable-detr_r50_16xb2-50e_pixray+loss.py', help='test config file path')
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\generalization\deform-detr\r50q300+head4\epoch_12.pth',
    #                     help='checkpoint file')
    # parser.add_argument('--config',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\generalization\deform-detr\r50q300base\deformable-detr_r50_16xb2-50e_pixray+loss.py', help='test config file path')
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\generalization\deform-detr\r50q300base\epoch_12.pth',
    #                     help='checkpoint file')
    # 可视化采样点 dino q30
    # parser.add_argument('--config',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\r50_pixray\basline30q\AO-DETRv2_r50_pixray.py', help='test config file path')
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\r50_pixray\basline30q\epoch_12.pth',
    #                     help='checkpoint file')
    # 可视化采样点 dino q30 +loss
    # parser.add_argument('--config',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\r50_pixray\01\AO-DETRv2_r50_pixray.py', help='test config file path')
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\r50_pixray\01\epoch_12.pth',
    #                     help='checkpoint file')
    # 可视化采样点 deform q30 +loss
    # parser.add_argument('--config',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\generalization\deform-detr\r50q30+head4\deformable-detr_r50_16xb2-50e_pixray+loss.py', help='test config file path')
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\generalization\deform-detr\r50q30+head4\epoch_12.pth',
    #                     help='checkpoint file')
    # 可视化采样点 deform q30 +loss
    # parser.add_argument('--config',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\generalization\deform-detr\r50q30base\deformable-detr_r50_16xb2-50e_pixray+loss.py', help='test config file path')
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\generalization\deform-detr\r50q30base\epoch_12.pth',
    #                     help='checkpoint file')
    # AO-DETRv2 (DINO + MMCL) swinl q30 pixray
    # parser.add_argument('--config',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\generalization\deform-detr\r50q30base\deformable-detr_r50_16xb2-50e_pixray+loss.py', help='test config file path')
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\AO-DETRv2\generalization\deform-detr\r50q30base\epoch_12.pth',
    #                     help='checkpoint file')


    # deformable detr r50 q300 hixray
    # parser.add_argument('--config',default=r'D:\Projects\DINO_mmdet3\mmdetection\configs\deformable_detr\deformable-detr_r50_16xb2-50e_hixray.py', help='test config file path')
    # parser.add_argument('--checkpoint',default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\hixray-baseline\deform-detr-r50\epoch_6.pth',
    #                     help='checkpoint file')
    # # AO-DETR swinl hixray normaml_reszie
    # parser.add_argument('--config', default='D:\Projects\DINO_mmdet3\mmdetection\configs\dino\AO-DETR_swin-l_hixray_normal_resize.py', help='train config file path')
    # parser.add_argument('--checkpoint', default='E:\D2E\Projects\DINO_mmdet3\checkpoint\hixray-baseline\AO-DETR-swinl-q320normaml_reszie2\epoch_12.pth',
    #                     help='the dir to save logs and models')

    # t-SNE 用途 dino swin-l MMCLv2 pixray q300 107 # todo 对比106和107 修改文件保存路径，跑完记得屏蔽函数
    # parser.add_argument('--config', default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\MMCLv2\swinl_pixray\107\MMCLv2_swinl_pixray.py', help='train config file path')
    # parser.add_argument('--checkpoint', default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\MMCLv2\swinl_pixray\107\epoch_12.pth',
    #                     help='the dir to save logs and models')
    # sampling points swin-l mmclv2 pixray q30 108
    parser.add_argument('--config', default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\MMCLv2\swinl_pixray\108\MMCLv2_swinl_pixray.py', help='train config file path')
    parser.add_argument('--checkpoint', default=r'E:\D2E\Projects\DINO_mmdet3\checkpoint\MMCLv2\swinl_pixray\108\epoch_12.pth',
                        help='the dir to save logs and models')
    parser.add_argument('--cam', default=False)
    parser.add_argument('--visualization_sampling_point',default=False)  # todo 用来控制是否可视化采样点，需要将multi_scale_deform_attn.py中更改路径
    parser.add_argument(
        '--out',
        # default='./PklForConfusion/50/epoch12.pkl',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')

    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
        # '--show', action='store_false', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        # default= 'imgs/',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.model.cam= args.cam  # TODO 添加是否添加 cam的参数
    cfg.model.visualization_sampling_point = args.visualization_sampling_point
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
