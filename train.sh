#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
python2 tools/train_net.py \
    --cfg configs/med/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    OUTPUT_DIR data/med01001

# -------------med01001---------------
# train med dataset , pretrained on icdar 17
    
    