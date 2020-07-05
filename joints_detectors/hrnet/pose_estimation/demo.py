'''
使用yolov3作为pose net模型的前处理
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import ipdb;pdb=ipdb.set_trace
import numpy as np
from tqdm import tqdm
from utilitys import plot_keypoint, PreProcess

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
import config
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

from utils.transforms import *
from lib.core.inference import get_final_preds
import cv2
#  import dataset
import models



def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        #  default='experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml',
                        default='experiments/coco/hrnet/w48_256x192_adam_lr1e-3.yaml',
                        #  default='experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument("-i", "--img_input", help="input video file name", default='/home/xyliu/Pictures/pose/soccer.png')
    parser.add_argument("-o", "--img_output", help="output video file name", default="output/result.png")
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    args.flip_test = True
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file



##### load model
def model_load(config):
    pdb()
    # lib/models/pose_hrnet.py:get_pose_net
    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )
    #  model_file_name  = 'models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth'
    model_file_name  = 'models/pytorch/pose_coco/pose_hrnet_w48_256x192.pth'
    #  model_file_name  = 'models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth'
    state_dict = torch.load(model_file_name)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k # remove module.
        #  print(name,'\t')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model



def main():
    args = parse_args()
    update_config(cfg, args)
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


    ########## 加载human detecotor model
    from lib.detector.yolo.human_detector import load_model as yolo_model
    human_model = yolo_model()

    from lib.detector.yolo.human_detector import main as yolo_det
    bboxs, scores = yolo_det(args.img_input, human_model)

    # bbox is coordinate location
    inputs, origin_img, center, scale = PreProcess(args.img_input, bboxs, scores, cfg)

    # load MODEL
    model = model_load(cfg)

    with torch.no_grad():
        # compute output heatmap
        inputs = inputs[:,[2,1,0]]
        output = model(inputs)
        # compute coordinate
        preds, maxvals = get_final_preds(
            cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

    image = plot_keypoint(origin_img, preds, maxvals, 0.3)
    cv2.imwrite(args.img_output, image)

if __name__ == '__main__':
    main()
