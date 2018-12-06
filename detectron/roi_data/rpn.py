# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Minibatch construction for Region Proposal Networks (RPN)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import numpy.random as npr

from detectron.core.config import cfg
import detectron.roi_data.data_utils as data_utils
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils

logger = logging.getLogger(__name__)


def get_rpn_blob_names(is_training=True):
    """Blob names used by RPN."""
    # im_info: (height, width, image scale)
    blob_names = ['im_info']
    if is_training:
        # gt boxes: (batch_idx, x1, y1, x2, y2, cls)
        blob_names += ['roidb']
        if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_RPN:
            # Same format as RPN blobs, but one per FPN level
            for lvl in range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1):
                # blob_names += [
                #     'rpn_labels_int32_wide_fpn' + str(lvl),
                #     'rpn_bbox_targets_wide_fpn' + str(lvl),
                #     'rpn_bbox_inside_weights_wide_fpn' + str(lvl),
                #     'rpn_bbox_outside_weights_wide_fpn' + str(lvl)
                # ]

                blob_names += [
                    'adarpn_labels_int32_wide_fpn' + str(lvl),
                    'adarpn_bbox_wh_wide_fpn' + str(lvl),
                    'adarpn_bbox_delta_wide_fpn' + str(lvl),
                    'adarpn_bbox_inside_weights_wide_fpn' + str(lvl),
                    'adarpn_bbox_outside_weights_wide_fpn' + str(lvl)
                ]
        else:
            # Single level RPN blobs
            blob_names += [
                'rpn_labels_int32_wide',
                'rpn_bbox_targets_wide',
                'rpn_bbox_inside_weights_wide',
                'rpn_bbox_outside_weights_wide'
            ]
    return blob_names


def add_rpn_blobs(blobs, im_scales, roidb):
    """Add blobs needed training RPN-only and end-to-end Faster R-CNN models."""
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_RPN:
        # RPN applied to many feature levels, as in the FPN paper
        k_max = cfg.FPN.RPN_MAX_LEVEL
        k_min = cfg.FPN.RPN_MIN_LEVEL
        foas = []
        for lvl in range(k_min, k_max + 1):
            field_stride = 2.**lvl
            anchor_sizes = (cfg.FPN.RPN_ANCHOR_START_SIZE * 2.**(lvl - k_min), )
            anchor_aspect_ratios = cfg.FPN.RPN_ASPECT_RATIOS
            foa = data_utils.get_field_of_anchors(
                field_stride, anchor_sizes, anchor_aspect_ratios
            )
            foas.append(foa)
        all_anchors = np.concatenate([f.field_of_anchors for f in foas])
    else:
        foa = data_utils.get_field_of_anchors(
            cfg.RPN.STRIDE, cfg.RPN.SIZES, cfg.RPN.ASPECT_RATIOS
        )
        all_anchors = foa.field_of_anchors

    for im_i, entry in enumerate(roidb):
        scale = im_scales[im_i]
        im_height = np.round(entry['height'] * scale)
        im_width = np.round(entry['width'] * scale)
        gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0)
        )[0]
        gt_rois = entry['boxes'][gt_inds, :] * scale
        im_info = np.array([[im_height, im_width, scale]], dtype=np.float32)
        blobs['im_info'].append(im_info)

        # Add RPN targets
        if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_RPN:
            # RPN applied to many feature levels, as in the FPN paper
            # print(entry['image'])
            rpn_blobs = _get_rpn_blobs(
                im_height, im_width, foas, all_anchors, gt_rois
            )
            for i, lvl in enumerate(range(k_min, k_max + 1)):
                for k, v in rpn_blobs[i].items():
                    blobs[k + '_fpn' + str(lvl)].append(v)
        else:
            # Classical RPN, applied to a single feature level
            rpn_blobs = _get_rpn_blobs(
                im_height, im_width, [foa], all_anchors, gt_rois
            )
            for k, v in rpn_blobs.items():
                blobs[k].append(v)

    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)

    valid_keys = [
        'has_visible_keypoints', 'boxes', 'segms', 'seg_areas', 'gt_classes',
        'gt_overlaps', 'is_crowd', 'box_to_gt_ind_map', 'gt_keypoints'
    ]
    minimal_roidb = [{} for _ in range(len(roidb))]
    for i, e in enumerate(roidb):
        for k in valid_keys:
            if k in e:
                minimal_roidb[i][k] = e[k]
    blobs['roidb'] = blob_utils.serialize(minimal_roidb)

    # Always return valid=True, since RPN minibatches are valid by design
    return True


def _get_rpn_blobs(im_height, im_width, foas, all_anchors, gt_boxes):
    total_anchors = all_anchors.shape[0]
    straddle_thresh = cfg.TRAIN.RPN_STRADDLE_THRESH

    if straddle_thresh >= 0:
        # Only keep anchors inside the image by a margin of straddle_thresh
        # Set TRAIN.RPN_STRADDLE_THRESH to -1 (or a large value) to keep all
        # anchors
        inds_inside = np.where(
            (all_anchors[:, 0] >= -straddle_thresh) &
            (all_anchors[:, 1] >= -straddle_thresh) &
            (all_anchors[:, 2] < im_width + straddle_thresh) &
            (all_anchors[:, 3] < im_height + straddle_thresh)
        )[0]
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
    else:
        inds_inside = np.arange(all_anchors.shape[0])
        anchors = all_anchors
    num_inside = len(inds_inside)

    logger.debug('total_anchors: {}'.format(total_anchors))
    logger.debug('inds_inside: {}'.format(num_inside))
    logger.debug('anchors.shape: {}'.format(anchors.shape))

    # Compute anchor labels:
    # label=1 is positive, 0 is negative, -1 is don't care (ignore)
    labels = np.empty((num_inside, ), dtype=np.int32)
    labels.fill(-1)
    if len(gt_boxes) > 0:
        # Compute overlaps between the anchors and the gt boxes overlaps
        anchor_by_gt_overlap = box_utils.bbox_overlaps(anchors, gt_boxes)
        # Map from anchor to gt box that has highest overlap
        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
        # For each anchor, amount of overlap with most overlapping gt box
        anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_inside),
                                                anchor_to_gt_argmax]

        # Map from gt box to an anchor that has highest overlap
        gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
        # For each gt box, amount of overlap with most overlapping anchor
        gt_to_anchor_max = anchor_by_gt_overlap[
            gt_to_anchor_argmax,
            np.arange(anchor_by_gt_overlap.shape[1])
        ]
        # Find all anchors that share the max overlap amount
        # (this includes many ties)
        anchors_with_max_overlap = np.where(
            anchor_by_gt_overlap == gt_to_anchor_max
        )[0]

        # Fg label: for each gt use anchors with highest overlap
        # (including ties)
        labels[anchors_with_max_overlap] = 1
        # Fg label: above threshold IOU
        labels[anchor_to_gt_max >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCH_SIZE_PER_IM)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False
        )
        labels[disable_inds] = -1
    fg_inds = np.where(labels == 1)[0]

    print('---------')
    print(len(fg_inds))
    # input()

    # subsample negative labels if we have too many
    # (samples with replacement, but since the set of bg inds is large most
    # samples will not have repeats)
    num_bg = cfg.TRAIN.RPN_BATCH_SIZE_PER_IM - np.sum(labels == 1)
    bg_inds = np.where(anchor_to_gt_max < cfg.TRAIN.RPN_NEGATIVE_OVERLAP)[0]
    if len(bg_inds) > num_bg:
        enable_inds = bg_inds[npr.randint(len(bg_inds), size=num_bg)]
        labels[enable_inds] = 0
    bg_inds = np.where(labels == 0)[0]

    bbox_targets = np.zeros((num_inside, 4), dtype=np.float32)
    bbox_targets[fg_inds, :] = data_utils.compute_targets(
        anchors[fg_inds, :], gt_boxes[anchor_to_gt_argmax[fg_inds], :]
    )

    # Bbox regression loss has the form:
    #   loss(x) = weight_outside * L(weight_inside * x)
    # Inside weights allow us to set zero loss on an element-wise basis
    # Bbox regression is only trained on positive examples so we set their
    # weights to 1.0 (or otherwise if config is different) and 0 otherwise
    bbox_inside_weights = np.zeros((num_inside, 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = (1.0, 1.0, 1.0, 1.0)

    # The bbox regression loss only averages by the number of images in the
    # mini-batch, whereas we need to average by the total number of example
    # anchors selected
    # Outside weights are used to scale each element-wise loss so the final
    # average over the mini-batch is correct
    bbox_outside_weights = np.zeros((num_inside, 4), dtype=np.float32)
    # uniform weighting of examples (given non-uniform sampling)
    num_examples = np.sum(labels >= 0)
    bbox_outside_weights[labels == 1, :] = 1.0 / num_examples
    bbox_outside_weights[labels == 0, :] = 1.0 / num_examples

    # Map up to original set of anchors
    labels = data_utils.unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = data_utils.unmap(
        bbox_targets, total_anchors, inds_inside, fill=0
    )
    bbox_inside_weights = data_utils.unmap(
        bbox_inside_weights, total_anchors, inds_inside, fill=0
    )
    bbox_outside_weights = data_utils.unmap(
        bbox_outside_weights, total_anchors, inds_inside, fill=0
    )

    # Split the generated labels, etc. into labels per each field of anchors
    blobs_out = []
    start_idx = 0
    for foa in foas:
        H = foa.field_size
        W = foa.field_size
        A = foa.num_cell_anchors
        end_idx = start_idx + H * W * A
        _labels = labels[start_idx:end_idx]
        _bbox_targets = bbox_targets[start_idx:end_idx, :]
        _bbox_inside_weights = bbox_inside_weights[start_idx:end_idx, :]
        _bbox_outside_weights = bbox_outside_weights[start_idx:end_idx, :]
        start_idx = end_idx

        # labels output with shape (1, A, height, width)
        _labels = _labels.reshape((1, H, W, A)).transpose(0, 3, 1, 2)
        # bbox_targets output with shape (1, 4 * A, height, width)
        _bbox_targets = _bbox_targets.reshape(
            (1, H, W, A * 4)).transpose(0, 3, 1, 2)
        # bbox_inside_weights output with shape (1, 4 * A, height, width)
        _bbox_inside_weights = _bbox_inside_weights.reshape(
            (1, H, W, A * 4)).transpose(0, 3, 1, 2)
        # bbox_outside_weights output with shape (1, 4 * A, height, width)
        _bbox_outside_weights = _bbox_outside_weights.reshape(
            (1, H, W, A * 4)).transpose(0, 3, 1, 2)
        blobs_out.append(
            dict(
                rpn_labels_int32_wide=_labels,
                rpn_bbox_targets_wide=_bbox_targets,
                rpn_bbox_inside_weights_wide=_bbox_inside_weights,
                rpn_bbox_outside_weights_wide=_bbox_outside_weights
            )
        )
    return blobs_out[0] if len(blobs_out) == 1 else blobs_out

# ------------------------------------------------------------------------------------

def add_adarpn_blobs(blobs, im_scales, roidb):
    """only support fpn manner."""
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_RPN:
        # RPN applied to many feature levels, as in the FPN paper
        k_max = cfg.FPN.RPN_MAX_LEVEL
        k_min = cfg.FPN.RPN_MIN_LEVEL
        anchor_points = []
        for lvl in range(k_min, k_max + 1):
            field_stride = 2.**lvl
            
            # 1, got all anchor center points
            fpn_max_size = cfg.FPN.COARSEST_STRIDE * np.ceil(
                cfg.TRAIN.MAX_SIZE / float(cfg.FPN.COARSEST_STRIDE))
            field_size = int(np.ceil(fpn_max_size / float(field_stride)))
            shifts = np.arange(0, field_size) * field_stride
            shift_x, shift_y = np.meshgrid(shifts, shifts)
            shift_x = shift_x.ravel()
            shift_y = shift_y.ravel()
            shifts = np.vstack((shift_x, shift_y)).transpose()

            center_x = (field_stride - 1) * 0.5
            center_y = center_x
            anchor_point = shifts + [center_x, center_y]
            anchor_points.append(anchor_point)

    for im_i, entry in enumerate(roidb):
        target_sum = 0
        scale = im_scales[im_i]
        im_height = np.round(entry['height'] * scale)
        im_width = np.round(entry['width'] * scale)
        gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0)
        )[0]
        gt_rois = entry['boxes'][gt_inds, :] * scale
        im_info = np.array([[im_height, im_width, scale]], dtype=np.float32)
        blobs['im_info'].append(im_info)

        # Add RPN targets
        if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_RPN:
            # RPN applied to many feature levels, as in the FPN paper
            # rpn_blobs = _get_adarpn_blobs(
            #     im_height, im_width, foas, all_anchors, gt_rois
            # )

            # for each fpn level, compute the target
            for i, lvl in enumerate(range(k_min, k_max + 1)):
                anchor_size = (cfg.FPN.RPN_ANCHOR_START_SIZE * 2.**(lvl - k_min), )
                this_level_ap = anchor_points[i]
                this_level_label = np.empty((this_level_ap.shape[0], ), dtype=np.int32)
                this_level_label.fill(-1)
                this_level_wh = np.zeros((this_level_ap.shape[0], 2), dtype=np.float32)
                this_level_box_delta = np.zeros((this_level_ap.shape[0], 4), dtype=np.float32)
                if len(gt_rois) > 0:
                    for ap_idx in range(len(this_level_ap)):
                        valid_gts = ap_inside_gt(gt_rois, this_level_ap[ap_idx])
                        if(len(valid_gts)>0):
                            scores, whs = find_best_box(this_level_ap[ap_idx], valid_gts, anchor_size[0]*1.0)
                            # print(whs)
                            areas = whs[:,0]*whs[:,1]
                            valid_idx = np.where((areas>0.25) & (areas < 4) 
                                & (scores >= cfg.TRAIN.RPN_POSITIVE_OVERLAP))[0]
                            if(valid_idx.shape[0]>0):
                                scores = scores[valid_idx]
                                whs = whs[valid_idx,:]
                                # fg_labels
                                this_level_label[ap_idx] = 1
                                # generally, when you set ROI overlap to 0.7 or much higher, for one anchor point only match one gt 
                                # box. so it will be ok to use the first box
                                this_level_wh[ap_idx] = whs[0]

                                # compute bbox_delata_target
                                gt_box = gt_rois[valid_idx[0]]

                                gt_width = gt_box[2] - gt_box[0] + 1.0
                                gt_height = gt_box[3] - gt_box[1] + 1.0
                                gt_ctr_x = gt_box[0] + 0.5 * gt_width
                                gt_ctr_y = gt_box[1] + 0.5 * gt_height

                                ex_width = whs[0][0] * anchor_size[0]
                                ex_height = whs[0][1] * anchor_size[0]

                                targets_dx = (gt_ctr_x - this_level_ap[ap_idx][0]) / ex_width
                                targets_dy = (gt_ctr_y - this_level_ap[ap_idx][1]) / ex_height
                                targets_dw = np.log(gt_width / ex_width)
                                targets_dh = np.log(gt_height / ex_height)

                                this_level_box_delta[ap_idx, :] = np.array([targets_dx, targets_dy, targets_dw, targets_dh])
                                
                                # print(this_level_ap[ap_idx])
                                # print(scores[0])
                                # print([gt_ctr_x, gt_ctr_y])
                                # print([ex_width, ex_height])
                                # print(gt_box)
                                # print(this_level_box_delta[ap_idx,:])
                                # input()
               
                fg_idx = np.where(this_level_label==1)[0]
                fg_num = len(fg_idx)
                
                # compute gb labels
                bg_inds = np.where(this_level_label==-1)[0]
                if len(bg_inds) > fg_num:
                    # need to filter some bg, so selecet labels a little more
                    enable_inds = bg_inds[npr.randint(len(bg_inds), size=(int(fg_num*1.2)))]
                    # filter out high IOU bg
                    keep=[]
                    for idx in range(len(enable_inds)):
                        iou, bg_whs = find_best_box(this_level_ap[enable_inds[idx]], gt_rois, anchor_size[0]*1.0)
                        bg_areas = bg_whs[:,0]*bg_whs[:,1]
                        bg_valid_idx = np.where( (bg_areas<0.2) | (bg_areas > 5) | (iou <= cfg.TRAIN.RPN_NEGATIVE_OVERLAP) )[0]
                        if bg_valid_idx.shape[0] == len(gt_rois):
                            keep.append(enable_inds[idx])
                    this_level_label[enable_inds] = 0
                else:
                    this_level_label[bg_inds] = 0

                bg_idx = np.where(this_level_label==0)[0]
                bg_num = len(bg_idx)

                this_level_box_inside_weight = np.zeros((this_level_ap.shape[0], 4), dtype=np.float32)
                this_level_box_inside_weight[this_level_label == 1, :] = (1.0, 1.0, 1.0, 1.0)

                this_level_box_outside_weight = np.zeros((this_level_ap.shape[0], 4), dtype=np.float32)
                # uniform weighting of examples (given non-uniform sampling)
                num_examples = fg_num + bg_num
                # print(num_examples)
                if num_examples > 0:
                    this_level_box_outside_weight[this_level_label == 1, :] = 1.0 / num_examples
                    this_level_box_outside_weight[this_level_label == 0, :] = 1.0 / num_examples


                # reshape as blob shape
                field_stride = 2.**lvl
                fpn_max_size = cfg.FPN.COARSEST_STRIDE * np.ceil(
                    cfg.TRAIN.MAX_SIZE / float(cfg.FPN.COARSEST_STRIDE))
                field_size = int(np.ceil(fpn_max_size / float(field_stride)))
                H = field_size
                W = field_size

                this_level_label= this_level_label.reshape((1, H, W, 1)).transpose(0,3,1,2)
                this_level_wh = this_level_wh.reshape((1, H, W, 2)).transpose(0,3,1,2)
                this_level_box_delta= this_level_box_delta.reshape((1, H, W, 4)).transpose(0,3,1,2)
                this_level_box_inside_weight= this_level_box_inside_weight.reshape((1, H, W, 4)).transpose(0,3,1,2)
                this_level_box_outside_weight= this_level_box_outside_weight.reshape((1, H, W, 4)).transpose(0,3,1,2)
                
                # add into blobs
                blobs['adarpn_labels_int32_wide_fpn' + str(lvl)].append(this_level_label)
                blobs['adarpn_bbox_wh_wide_fpn' + str(lvl)].append(this_level_wh)
                blobs['adarpn_bbox_delta_wide_fpn' + str(lvl)].append(this_level_box_delta)
                blobs['adarpn_bbox_inside_weights_wide_fpn' + str(lvl)].append(this_level_box_inside_weight)
                blobs['adarpn_bbox_outside_weights_wide_fpn' + str(lvl)].append(this_level_box_outside_weight)
                        
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)

    valid_keys = [
        'has_visible_keypoints', 'boxes', 'segms', 'seg_areas', 'gt_classes',
        'gt_overlaps', 'is_crowd', 'box_to_gt_ind_map', 'gt_keypoints'
    ]
    minimal_roidb = [{} for _ in range(len(roidb))]
    for i, e in enumerate(roidb):
        for k in valid_keys:
            if k in e:
                minimal_roidb[i][k] = e[k]
    blobs['roidb'] = blob_utils.serialize(minimal_roidb)

    # Always return valid=True, since RPN minibatches are valid by design
    return True

def ap_inside_gt(gts, ap):
    valid_gts = []
    for gt_idx in range(len(gts)):
        if ap[0] >= gts[gt_idx][0] and ap[0] <= gts[gt_idx][2] \
            and ap[1] >= gts[gt_idx][1] and ap[1] <= gts[gt_idx][3]:
            valid_gts.append(gts[gt_idx])
    return valid_gts

def find_best_box(ap, gt, norm=1.0):
    def aim_func(x, gt, c):
        gtw = gt[2] - gt[0] + 1
        gth = gt[3] - gt[1] + 1

        # obj = [c(1)-x(1)/2, c(2)-x(2)/2, c(1)+x(1)/2, c(2) + x(2)/2]
        obj = [c[0]-x[0]*0.5, c[1]-x[1]*0.5, c[0]+x[0]*0.5-1, c[1]+x[1]*0.5-1]
        objw = obj[2] - obj[0] + 1
        objh = obj[3] - obj[1] + 1

        xx1 = max(gt[0], obj[0])
        yy1 = max(gt[1], obj[1])
        xx2 = min(gt[2], obj[2])
        yy2 = min(gt[3], obj[3])

        h = max(random.random()/10, yy2-yy1)
        w = max(random.random()/10, xx2-xx1)

        I = w*h
        IOU = I/((gtw * gth + objw * objh - I)*1.0)
        return -IOU
    
    from scipy.optimize import minimize
    import random
    scores=np.zeros((len(gt),), dtype=np.float32)
    whs=np.zeros((len(gt),2), dtype=np.float32)
    # return scores, whs
    for gt_idx in range(len(gt)):
        wh = max(abs(gt[gt_idx][2]-gt[gt_idx][0]), abs(gt[gt_idx][3]-gt[gt_idx][1]))
        x0 = (int(wh*2), int(wh*2))
        res = minimize(aim_func, args=(gt[gt_idx], ap), x0=x0, method='Nelder-Mead', tol=1e-9)
        scores[gt_idx] = -res.fun
        whs[gt_idx] = res.x/(norm*1.0)
    return scores, whs
