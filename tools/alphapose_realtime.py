import os
import cv2
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
import ipdb;pdb=ipdb.set_trace
import time

from joints_detectors.hrnet.pose_estimation.video import getTwoModel, getKptsFromImage
bboxModel, poseModel = getTwoModel()
interface2D = getKptsFromImage
from tools.utils import videopose_model_load as Model3Dload
model3D = Model3Dload()
from tools.utils import interface as VideoPoseInterface
interface3D = VideoPoseInterface
from tools.utils import draw_3Dimg, draw_2Dimg, videoInfo,resize_img

def main(VideoName):
    cap, cap_length = videoInfo(VideoName)
    # cap=cv2.VideoCapture(0)
    kpt2Ds = []
    queueSize = 30
    for i in tqdm(range(cap_length)):
    # i=0
    # while(True):
        _, frame = cap.read()
        frame, W, H = resize_img(frame)

        try:
            t0 = time.time()
            joint2D = interface2D(bboxModel, poseModel, frame)
        except Exception as e:
            print(e)
            continue

        if i == 0:
            for _ in range(queueSize):
                kpt2Ds.append(joint2D)
        elif i < queueSize:
            kpt2Ds.append(joint2D)
            kpt2Ds.pop(0)
        else:
            kpt2Ds.append(joint2D)

        joint3D = interface3D(model3D, np.array(kpt2Ds), W, H)
        joint3D_item = joint3D[-1] #(17, 3)
        draw_3Dimg(joint3D_item, frame, display=1, kpt2D=joint2D)

        # i = i+1
        if cv2.waitKey(1) & 0xff == ord('q'):
            cap.release()
            break
        print('total comsume {:0.3f} s'.format(time.time() - t0))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-video", "--video_input", help="input video file name", default="/home/xyliu/Videos/sports/dance.mp4")
    args = parser.parse_args()
    VideoName = args.video_input
    print('Input Video Name is ', VideoName)
    main(VideoName)


# python tools/hrnet_realtime.py -video /path/to/video.mp4