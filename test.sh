#! /bin/bash
export CUDA_VISIBLE_DEVICES=1
python2 tools/test_net.py \
    --cfg configs/icdar/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    TEST.WEIGHTS ~/3-deepLearning/3-scene_text_detection/detectron/model/icdar/adarpn2/train/icdar_2015_train/generalized_rcnn/model_iter4999.pkl \
    NUM_GPUS 1 \
    VIS True
