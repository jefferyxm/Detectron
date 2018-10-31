python2 tools/infer_simple.py \
    --cfg configs/icdar/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    --output-dir /home/xiem/tmp/detectron-visualizations \
    --image-ext jpg \
    --output-ext jpg \
    --wts /home/xiem/tmp/detectron-output_02/train/icdar_2015_train/generalized_rcnn/model_final.pkl \
    demo