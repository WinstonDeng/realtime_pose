# Realtime_pose
Realtime 2d pose using camera.

Pose estimator: HRNet

# Environment
pytorch >= 1.0

python >= 3.6

Anaconda3

run this create virtural env:

> conda env create -f env_info_file.yml

# Usage

> python tools/hrnet_2d_realtime.py

# Pre-trained model

- HRNet:

1. load [pose model](https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA) in

   joints_detectors/hrnet/models/pytorch/pose_coco/ 

2. load [yolov3](https://drive.google.com/file/d/1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC/view) in

   joints_detectors/hrnet/lib/detector/yolo/
   
   or download by 
   
   > wget https://pjreddie.com/media/files/yolov3.weights

- AlphaPose:

1. load [due_se](https://drive.google.com/file/d/1OPORTWB2cwd5YTVBX-NE8fsauZJWsrtW/view) in

   joints_detectors/Alphapose/models/sppe/

2. load [yolov3](https://drive.google.com/file/d/1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC/view) (same to the yolov3 file in HRNet dir) in

   joints_detectors/Alphapose/models/yolo/ 
   
# Reference
Modified from [https://github.com/lxy5513/videopose](https://github.com/lxy5513/videopose)
