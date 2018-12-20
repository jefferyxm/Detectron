export CUDA_VISIBLE_DEVICES=1
python2 tools/infer_simple.py \
    --cfg configs/icdar/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    --output-dir /home/xiem/tmp/detectron-visualizations \
    --image-ext jpg \
    --output-ext png \
    --wts ~/3-deepLearning/3-scene_text_detection/detectron/model/icdar/adarpn/train/icdar_2015_train/generalized_rcnn/model_iter44999.pkl \
    /home/xiem/3-deepLearning/3-scene_text_detection/detectron/detectron/datasets/data/icdar/icdar15/test