"""val a Detectron network on an val imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml

from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.core.config import get_output_dir
from detectron.core.rpn_generator import generate_rpn_on_dataset
from detectron.core.rpn_generator import generate_rpn_on_range
from detectron.core.test import im_detect_all
from detectron.datasets import task_evaluation
from detectron.datasets.json_dataset import JsonDataset
from detectron.modeling import model_builder
from detectron.utils.io import save_object
from detectron.utils.timer import Timer
import detectron.utils.c2 as c2_utils
import detectron.utils.env as envu
import detectron.utils.net as net_utils
import detectron.utils.subprocess as subprocess_utils
import detectron.utils.vis as vis_utils

logger = logging.getLogger(__name__)

def create_val_model():
    """
    create validation model
    """
    model = model_builder.create(cfg.MODEL.TYPE, train=False)
    workspace.RunNetOnce(model.param_init_net)
    model_builder.add_inference_inputs(model)
    workspace.CreateNet(model.net)
    workspace.CreateNet(model.conv_body_net)
    if cfg.MODEL.MASK_ON:
        workspace.CreateNet(model.mask_net)
    if cfg.MODEL.KEYPOINTS_ON:
        workspace.CreateNet(model.keypoint_net)
    return model


def get_val_dataset():
    # suppose only one validation dataset
    dataset_name = cfg.TEST.DATASETS[0]
    dataset = JsonDataset(dataset_name)
    roidb = dataset.get_roidb()
    return dataset, roidb


def validation_dataset(model, roidb, gpu_id=0):
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES
    all_boxes, all_segms, all_keyps = empty_results(num_classes, num_images)
    timers = defaultdict(Timer)
    for i, entry in enumerate(roidb):
        im = cv2.imread(entry['image'])
        with c2_utils.NamedCudaScope(gpu_id):
            cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(
                model, im, None, timers
            )

        extend_results(i, all_boxes, cls_boxes_i)
        if cls_segms_i is not None:
            extend_results(i, all_segms, cls_segms_i)
        if cls_keyps_i is not None:
            extend_results(i, all_keyps, cls_keyps_i)

        if i % 10 == 0:  # Reduce log file size
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            eta_seconds = ave_total_time * (num_images - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            det_time = (
                timers['im_detect_bbox'].average_time +
                timers['im_detect_mask'].average_time +
                timers['im_detect_keypoints'].average_time
            )
            misc_time = (
                timers['misc_bbox'].average_time +
                timers['misc_mask'].average_time +
                timers['misc_keypoints'].average_time
            )
            logger.info(
                (
                    'im_detect: {:d}/{:d} {:.3f}s + {:.3f}s (eta: {})'
                ).format(
                    i + 1, num_images, det_time, misc_time, eta
                )
            )
    return all_boxes, all_segms, all_keyps

def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_keyps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes, all_segms, all_keyps


def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for cls_idx in range(1, len(im_res)):
        all_res[cls_idx][index] = im_res[cls_idx]


def run_validation(val_model, val_dataset, val_roidb, cur_iter, val_output_dir, tb_logger):
    all_boxes, all_segms, all_keyps = validation_dataset(val_model, val_roidb)
    val_results = task_evaluation.evaluate_all(
            val_dataset, all_boxes, all_segms, all_keyps, val_output_dir)
    
    # write the result to tensorboard
    tb_logger.write_scalars(val_results[cfg.TEST.DATASETS[0]]['box'], cur_iter)
    tb_logger.write_scalars(val_results[cfg.TEST.DATASETS[0]]['mask'], cur_iter)

    # get AP and AR100 to compute Hmean
    box_AP = val_results[cfg.TEST.DATASETS[0]]['box']['AP']
    box_AR100 = val_results[cfg.TEST.DATASETS[0]]['box']['AR100']

    mask_AP = val_results[cfg.TEST.DATASETS[0]]['mask']['AP']
    mask_AR100 = val_results[cfg.TEST.DATASETS[0]]['mask']['AR100']
    
    if box_AP==0 or box_AR100==0:
        box_hmean=0
    else:
        box_hmean = (2*box_AP*box_AR100*1.0)/((box_AP+box_AR100)*1.0)

    if mask_AP==0 or mask_AR100==0:
        mask_hmean=0
    else:
        mask_hmean = (2*mask_AP*mask_AR100*1.0)/((mask_AP+mask_AR100)*1.0)
    
    hmean = 0.5*(box_hmean+mask_hmean)

    tb_logger.write_scalars(dict(box_Hmean=box_hmean,
                                    mask_Hmean=mask_hmean,
                                    Hmean=hmean), cur_iter)
    return hmean
