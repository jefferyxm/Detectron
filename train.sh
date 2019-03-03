#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
python2 tools/train_net.py \
    --cfg configs/icdar/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    OUTPUT_DIR data/debug

# -------------exp01002---------------
# train on icdar 2017

# ------------exp01003----------------
# finetune on icdar2015
    
    