import os
import cv2

import threading
from PyQt5.QtCore import QFile
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap

from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
import ipdb;pdb=ipdb.set_trace
import time

from joints_detectors.hrnet.pose_estimation.video import getTwoModel, getKptsFromImage
bboxModel, poseModel = getTwoModel()
interface2D = getKptsFromImage

from tools.utils import draw_2Dimg, resize_img


class Display:
    def __init__(self, ui, mainWindow):
        self.ui = ui
        self.mainWindow = mainWindow

        # set default video source as camera stream
        self.ui.radioButtonCam.setChecked(True)
        self.isCamera = True

        # set signal 
        ui.Open.clicked.connect(self.Open)
        ui.Close.clicked.connect(self.Close)
        ui.radioButtonCam.clicked.connect(self.radioButtonCam)
        ui.radioButtonFile.clicked.connect(self.radioButtonFile)

        # create a close event defaulted by stop
        self.stopEvent = threading.Event()
        self.stopEvent.clear()

    def radioButtonCam(self):
        self.isCamera = True
    
    def radioButtonFile(self):
        self.isCamera = False

    def Open(self):
        if not self.isCamera:
            # todo video length
            self.fileName, self.fileType = QFileDialog.getOpenFileName(self.mainWindow, 'Choose file', '', '*.mp4')
            self.cap = cv2.VideoCapture(self.fileName)
            self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.cap = cv2.VideoCapture(0)

        th = threading.Thread(target=self.Display)
        th.start()
    
    def Close(self):
        self.stopEvent.set()

    def Display(self):
        self.ui.Open.setEnabled(False)
        self.ui.Close.setEnabled(True)

        while self.cap.isOpened():
            success, frame = self.cap.read()
            frame, W, H = resize_img(frame)
            try:
                # get 2d pose keypoints from image using pose estimator (hrnet)
                joint2D = interface2D(bboxModel, poseModel, frame)
 
            except Exception as e:
                print(e)
                continue

            # draw pose keypoints into source image
            img = draw_2Dimg(frame, joint2D, None)
        
            # RGB to BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_out = QImage(img_bgr.data, W, H, QImage.Format_RGB888)
            self.ui.DisplayLabel.setPixmap(QPixmap.fromImage(img_out))

            if self.isCamera:
                cv2.waitKey(1)
            else:
                cv2.waitKey(int(1000/self.frameRate))
            
            if self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                self.ui.DisplayLabel.clear()
                self.ui.Close.setEnabled(False)
                self.ui.Open.setEnabled(True)
                self.cap.release()
                break

def show_ui():
    import sys
    import DisplayUI
    from PyQt5.QtWidgets import QApplication, QMainWindow
    
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = DisplayUI.Ui_MainWindow()

    ui.setupUi(mainWindow)
    display = Display(ui, mainWindow)
    mainWindow.show()

    sys.exit(app.exec_())


def show_cv2():
    # use camera
    cap=cv2.VideoCapture(0)
    while True:
        # read every frame from camera
        _, frame = cap.read()
        frame, W, H = resize_img(frame)

        try:
            t0 = time.time()
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
        # print('total comsume {:0.3f} s'.format(time.time() - t0))


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    # show_cv2()
    show_ui()
