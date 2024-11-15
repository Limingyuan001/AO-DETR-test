**说明**

想要运行AO-DETR只需要运行tools/train_ao-detr.py

DINO和其他Deformable DETR-based variants可以参考train_ao-detr.py中的注释！

PIXray，OPIXray，HIXray数据集都可以运行，只需要更换config文件即可

🎉️ 🎉️ 🎉️

**本工作的论文《### [AO-DETR: Anti-Overlapping DETR for X-Ray Prohibited Items Detection](https://ieeexplore.ieee.org/document/10746383/)》已经录用在TNNLS期刊上，目前处于Early Access状态！！！
如果我的工作对您有帮助的话，欢迎使用以下BibTex进行引用：**

@ARTICLE{10746383,
author={Li, Mingyuan and Jia, Tong and Wang, Hao and Ma, Bowen and Lu, Hui and Lin, Shuyang and Cai, Da and Chen, Dongyue},
journal={IEEE Transactions on Neural Networks and Learning Systems},
title={AO-DETR: Anti-Overlapping DETR for X-Ray Prohibited Items Detection},
year={2024},
volume={},
number={},
pages={1-15},
keywords={X-ray imaging;Detectors;Feature extraction;Decoding;Accuracy;Training;Inspection;Transformers;Security;Proposals;Iterative refinement boxes;label assignment;object detection;X-ray inspection},
doi={10.1109/TNNLS.2024.3487833}}

🚀️ 🚀️ 🚀️

**PIXray in COCO format 独家链接！！！**

由于原版PIXray数据集的格式并不规范，无法直接使用在主流代码库。因此，我整理了一份coco格式的PIXray数据集，可以通过连接下载[PIXray_coco](https://drive.google.com/drive/folders/1jkLaB1YVMaxDZ6Qv84ad5zHIXd80thAr?usp=sharing)
I have put together a copy of the PIXray dataset in coco format, which can be downloaded via connection[PIXray_coco](https://drive.google.com/drive/folders/1jkLaB1YVMaxDZ6Qv84ad5zHIXd80thAr?usp=sharing)

**Requirements！**

python 3.9.17

pytorch 1.13.1

mmdet3.1.0

mmcv 2.0.1

具体安装教程参见[mmdetection官网教程](https://mmdetection.readthedocs.io/en/v3.1.0/get_started.html)

