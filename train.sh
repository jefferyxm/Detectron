#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
python2 tools/train_net.py \
    --cfg configs/icdar/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    OUTPUT_DIR data/up_ada_exp03


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

# --------------up_ada_exp01----------------
# 1, using icdar evaluation tools
# 2, up sampling rpn blobs to get fine result

# --------------up_ada_exp02----------------
# RPN_PRE_NMS_TOP_N = 8000 / 2000
# RPN_POST_NMS_TOP_N = 2000 / 1000

# --------------up_ada_exp03----------------
# based on up_ada_exp02
# do not use focal loss on rpn
# verify the focal loss


    