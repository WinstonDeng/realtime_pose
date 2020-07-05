import torch
#  import ipdb;ipdb.set_trace()
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import sys

main_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(main_path)

import sys
sys.path.append("../")
from opt import opt

from dataloader import VideoLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

import ntpath
from tqdm import tqdm
import time
from fn import getTime
import cv2

from pPose_nms import pose_nms, write_json

args = opt
args.dataset = 'coco'
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


def model_load():
    return model

def image_interface(model, image):
    pass

def handle_video(videofile):
    args.video = videofile
    videofile = args.video
    mode = args.mode

    if not len(videofile):
        raise IOError('Error: must contain --video')

    # Load input video
    data_loader = VideoLoader(videofile, batchSize=args.detbatch).start()
    (fourcc,fps,frameSize) = data_loader.videoinfo()

    print('the video is {} f/s'.format(fps))

    # Load detection loader
    print('Loading YOLO model..')
    sys.stdout.flush()
    det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start()
    #  start a thread to read frames from the file video stream
    det_processor = DetectionProcessor(det_loader).start()

    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Data writer
    save_path = os.path.join(args.outputpath, 'AlphaPose_'+ntpath.basename(videofile).split('.')[0]+'.avi')
    writer = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()

    im_names_desc =  tqdm(range(data_loader.length()))
    batchSize = args.posebatch
    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
            if orig_img is None:
                break
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                continue

            ckpt_time, det_time = getTime(start_time)
            runtime_profile['dt'].append(det_time)
            # Pose Estimation

            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].cuda()
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)

            hm = hm.cpu().data
            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])

            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)

        if args.profile:
            # TQDM
            im_names_desc.set_description(
            'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
            )

    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
    while(writer.running()):
        pass
    writer.stop()
    final_result = writer.results()

    # 获取第 0 个框的人
    kpts = []
    for i in range(len(final_result)):
        try:
            preds = final_result[i]['result']
            # preds[i]['keypoints'] (17,2)
            # preds[i]['kp_score'] (17,1)
            # preds[i]['proposal_score'] (1)
            # 选择 y 坐标最大的人 —— 用于打羽毛球视频
            max_index = 0
            min_index = 0
            # max_y = np.mean(preds[0]['keypoints'].data.numpy()[:, 1])
            min_x = np.mean(preds[0]['keypoints'].data.numpy()[:, 0])
            max_x = np.mean(preds[0]['keypoints'].data.numpy()[:, 0])

            for k in range(len(preds)):
                # tmp_y = np.mean(preds[k]['keypoints'].data.numpy()[:, 1])
                tmp_x = np.mean(preds[k]['keypoints'].data.numpy()[:, 0])
                # if tmp_y > max_y:
                if tmp_x < min_x:
                    min_index = k
                    # max_y = tmp_y
                    min_x = tmp_x
            for k in range(len(preds)):
                # tmp_y = np.mean(preds[k]['keypoints'].data.numpy()[:, 1])
                tmp_x = np.mean(preds[k]['keypoints'].data.numpy()[:, 0])
                # if tmp_y > max_y:
                if tmp_x > max_x:
                    max_index = k
                    max_x = tmp_x
            mid_index = 0
            for k in range(len(preds)):
                if k == max_index or k == min_index:
                    continue
                mid_index = k
            kpt = preds[mid_index]['keypoints']
            # kpt = final_result[i]['result'][0]['keypoints']
            kpts.append(kpt.data.numpy())

        except:
            # print(sys.exc_info())
            print('error...')

    filename = os.path.basename(args.video).split('.')[0]
    name = filename + '.npz'
    kpts = np.array(kpts).astype(np.float32)
    # print('kpts npz save in ', name)
    # np.savez_compressed(name, kpts=kpts)
    return kpts

if __name__ == "__main__":
    handle_video()
