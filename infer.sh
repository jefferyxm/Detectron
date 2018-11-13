export CUDA_VISIBLE_DEVICES=0
# python2 tools/infer_simple.py \
#     --cfg /home/xiem/3-deepLearning/3-scene_text_detection/detectron/configs/med/e2e_mask_rcnn_R-50-FPN_1x.yaml \
#     --output-dir tmp_res_test \
#     --image-ext jpg \
#     --output-ext jpg \
#     --wts /home/xiem/med_tmp/detectron-output/train/med_train/generalized_rcnn/model_iter4999.pkl \
#     demo/train_pic/

python2 tools/infer_simple.py \
    --cfg /home/xiem/3-deepLearning/3-scene_text_detection/detectron/configs/med/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    --output-dir tmp_res_test \
    --image-ext jpg \
    --output-ext jpg \
    --wts /home/xiem/med_tmp/detectron-output-rpn16/train/med_train/generalized_rcnn/model_iter5999.pkl \
    demo/train_pic/