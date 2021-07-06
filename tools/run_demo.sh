#!/bin/bash
python3 demo_novis.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt model/pv_rcnn_8369.pth \
    --data_path ../data/kitti/training/velodyne/000008.bin
