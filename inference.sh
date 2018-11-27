export CUDA_VISIBLE_DEVICES=1
python2 tools/infer_simple.py \
    --cfg configs/icdar/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    --output-dir /home/xiem/tmp/detectron-visualizations \
    --image-ext jpg \
    --output-ext jpg \
    --wts /home/xiem/icdar_tmp/detectron-output_01/train/icdar_2015_train/generalized_rcnn/model_final.pkl \
    /home/xiem/3-deepLearning/3-scene_text_detection/dataset/ic15/test
