#! /bin/bash
export CUDA_VISIBLE_DEVICES=1
python2 tools/train_net.py \
    --cfg configs/icdar/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    OUTPUT_DIR data/exp01


# ---------------exp01------------------- 
# retrain icdar15 with tblogger and validation
# want to find the best procedure of train