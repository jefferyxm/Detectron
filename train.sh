#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
python2 tools/train_net.py \
    --cfg configs/icdar/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    OUTPUT_DIR data/ada_exp05_2


# ---------------ada_exp01------------------- 
# reduce params

# ---------------ada_exp02------------------- 
# continue reduce some params

# ---------------ada_exp03------------------- 
# add params

# ---------------ada_exp04------------------- 
#  focal loss

# ---------------ada_exp05------------------- 
#  5_1: wh with weight
#  5_2: more weight on w/h < 0.67
    