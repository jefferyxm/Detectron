#! /bin/bash
export CUDA_VISIBLE_DEVICES=1
python2 tools/train_net.py \
    --cfg configs/icdar/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    OUTPUT_DIR ~/3-deepLearning/3-scene_text_detection/detectron/model/icdar/debug
