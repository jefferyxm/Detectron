#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
python2 tools/train_net.py \
    --cfg configs/icdar/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    OUTPUT_DIR data/ada_exp02


# ---------------ada_exp01------------------- 
# reduce params

# ---------------ada_exp02------------------- 
# continue reduce some params