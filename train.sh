#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
python2 tools/train_net.py \
    --cfg configs/icdar/P-R50-FPN.yaml \
    OUTPUT_DIR data/exp01004

# -------------exp01002---------------
# train on icdar 2017

# ------------exp01003----------------
# finetune on icdar2015
    
    