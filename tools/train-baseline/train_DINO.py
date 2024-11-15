# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')


    # dino swinl PIXray done 记得改LFD
    # parser.add_argument('--config', default='../../configs/dino/dinov2-5scale_swin-l_8xb2-12e_pixray_mine_q900_backbone.py',
    #                     help='train config file path')
    # parser.add_argument('--work-dir', default='E:\D2E\Projects\DINO_mmdet3\checkpoint/pixray-baseline/dino-swin-lbackbone2',
    #                     help='the dir to save logs and models')

    # dino swinl opixray
    # parser.add_argument('--config', default='../../configs/dino/DINO_swin-l_opixray.py',
    #                     help='train config file path')
    # parser.add_argument('--work-dir', default='E:\D2E\Projects\DINO_mmdet3\checkpoint/opixray-baseline/dino-swin-l_q10',
    #                     help='the dir to save logs and models')

    # dino r50 opixray
    # parser.add_argument('--config', default='../../configs/dino/DINO_r50_opixray.py.py', help='train config file path')
    # parser.add_argument('--work-dir', default='E:\D2E\Projects\DINO_mmdet3\checkpoint/opixray-baseline/dino-r50-net', help='the dir to save logs and models')

    # AO-DETR r50 opixray
    # parser.add_argument('--config', default='../../configs/dino/AO-DETR_r50_opixray.py', help='train config file path')
    # parser.add_argument('--work-dir', default='E:\D2E\Projects\DINO_mmdet3\checkpoint/opixray-baseline/AO-DETR-r50', help='the dir to save logs and models')
    # AO-DETR swin-l opixray
    # parser.add_argument('--config', default='../../configs/dino/AO-DETR_swin-l_opixray.py', help='train config file path')
    # parser.add_argument('--work-dir', default='E:\D2E\Projects\DINO_mmdet3\checkpoint/opixray-baseline/AO-DETR-swin-l-q10',
    #                     help='the dir to save logs and models')
    # todo fine-tune coco pretrain model to obtain the best model
    # AO-DETR r50 opixray
    # parser.add_argument('--config', default='../../configs/dino/AO-DETR_r50_opixray_fine-tune.py', help='train config file path')
    # parser.add_argument('--work-dir', default='E:\D2E\Projects\DINO_mmdet3\checkpoint/opixray-baseline/AO-DETR-r50_fine-tune', help='the dir to save logs and models')
    # AO-DETR swin-l opixray fine-tune 320
    # parser.add_argument('--config', default='../../configs/dino/AO-DETR_swin-l_opixray_fine-tune.py', help='train config file path')
    # parser.add_argument('--work-dir', default='E:\D2E\Projects\DINO_mmdet3\checkpoint/opixray-baseline/AO-DETR-swin-l-q10_fine-tune',
    #                     help='the dir to save logs and models')
    # AO-DETR swin-l opixray fine-tune 640 冲best
    # parser.add_argument('--config', default='../../configs/dino/AO-DETR_swin-l_opixray_fine-tune_640.py', help='train config file path')
    # parser.add_argument('--work-dir', default='E:\D2E\Projects\DINO_mmdet3\checkpoint/opixray-baseline/AO-DETR-swin-l-q100_fine-tune_640_e36',
    #                     help='the dir to save logs and models')
    # AO-DETR r50 opixray fine-tune coco 640 冲best
    # parser.add_argument('--config', default='../../configs/dino/AO-DETR_r50_opixray_fine-tune_640.py', help='train config file path')
    # parser.add_argument('--work-dir', default='E:\D2E\Projects\DINO_mmdet3\checkpoint/opixray-baseline/AO-DETR-r50-q100_fine-tune_640_e15',
    #                     help='the dir to save logs and models')

    # AO-DETR r50 hixray
    # parser.add_argument('--config', default='../../configs/dino/AO-DETR_r50_hixray.py', help='train config file path')
    # parser.add_argument('--work-dir', default='E:\D2E\Projects\DINO_mmdet3\checkpoint/hixray-baseline/AO-DETR-r50-q320', help='the dir to save logs and models')
    # AO-DETR r50 hixray normaml_reszie
    # parser.add_argument('--config', default='../../configs/dino/AO-DETR_r50_hixray_normaml_reszie.py', help='train config file path')
    # parser.add_argument('--work-dir', default='E:\D2E\Projects\DINO_mmdet3\checkpoint/hixray-baseline/AO-DETR-r50-q320normaml_reszie2',
    #                     help='the dir to save logs and models')
    # AO-DETR swin-l hixray
    # parser.add_argument('--config', default='../../configs/dino/AO-DETR_swin-l_hixray.py', help='train config file path')
    # parser.add_argument('--work-dir', default='E:\D2E\Projects\DINO_mmdet3\checkpoint/hixray-baseline/AO-DETR-swin-l', help='the dir to save logs and models')
    # AO-DETR swinl hixray normaml_reszie
    parser.add_argument('--config', default='../../configs/dino/AO-DETR_swin-l_hixray_normal_resize.py', help='train config file path')
    parser.add_argument('--work-dir', default='E:\D2E\Projects\DINO_mmdet3\checkpoint/hixray-baseline/AO-DETR-swinl-q320normaml_reszie2',
                        help='the dir to save logs and models')

    # DINO r50 hixray
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        # default=True,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
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
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
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

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
