#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
python2 tools/train_net.py \
    --cfg configs/icdar/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    OUTPUT_DIR data/exp02004


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

# --------------up_ada_exp04----------------
# cls using sigmoid

# -------------up_ada_exp04_2-----------------
# change the distribution of fast rcnn training samples
# original p:n = 1:4
# original p:n = 1:1

# -------------up_ada_exp04_3-----------------
# using adaptive feature pooling

# -------------up_ada_exp04_4-----------------
# vertical rotate imgage

# -------------pure_ada_anchor----------------
# do not use up sampling
# do not use afp
# do not use focal loss

# ------------pure_ada_anchor_2---------------
# do not rpn box regeression
# rpn_wh loss scaled 0.1 
# keep proposal with score >= 0.5 ranther keep top k

    # pure_ada_anchor_2_1
    # use golbal pool instead of fc6

    # pure_ada_anchor_2_2
    # add_roi_Xconv1fc_gn_head

# ------------pure_ada_anchor_5---------------
# trian with icdar 2017

# ------------exp02001-----------------------
# use icdar2017 pretrained model 

# ------------exp02003-----------------------
# add deform convolution and training on icdar 2017

# ------------exp02004------------------------
#  deform convolution on icdar 2015,
#  fintune on icdar 2017