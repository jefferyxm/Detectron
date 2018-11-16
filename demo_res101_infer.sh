export CUDA_VISIBLE_DEVICES=0
python tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir ~/icdar_tmp/detectron-visualizations \
    --image-ext jpg \
    --wts  ~/icdar_tmp/model_final.pkl \
    demo

