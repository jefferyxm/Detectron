#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
python2 tools/test_net.py \
    --cfg configs/icdar/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    TEST.WEIGHTS data/up_ada_exp04/train/model_iter19.pkl \
    NUM_GPUS 1 \
    # VIS True
