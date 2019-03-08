#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
python2 tools/test_net.py \
    --cfg configs//med/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    TEST.WEIGHTS data/med01001/train/model_best.pkl \
    NUM_GPUS 1 \
    VIS True
