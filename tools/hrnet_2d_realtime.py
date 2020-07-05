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

from tools.utils import draw_2Dimg, resize_img

def main():
    # use camera
    cap=cv2.VideoCapture(0)
    while True:
        # read every frame from camera
        _, frame = cap.read()
        frame, W, H = resize_img(frame)

        try:
            # get 2d pose keypoints from image using pose estimator (hrnet)
            joint2D = interface2D(bboxModel, poseModel, frame)
        except Exception as e:
            print(e)
            continue

        # draw pose keypoints into source image
        img = draw_2Dimg(frame, joint2D, None)
        cv2.imshow('result_view', img)

        # exit control
        if cv2.waitKey(1) & 0xff == ord('q'):
            cap.release()
            break


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    main()
