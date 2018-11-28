#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
python2 tools/train_net.py \
    --cfg configs/icdar/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    OUTPUT_DIR ~/icdar_tmp/2018_11_27_iou_0.5 
    
    
