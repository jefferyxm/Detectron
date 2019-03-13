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

"""Functions for using a Feature Pyramid Network (FPN)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import numpy as np

from detectron.core.config import cfg
from detectron.modeling.generate_anchors import generate_anchors
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.modeling.ResNet as ResNet
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils

# Lowest and highest pyramid levels in the backbone network. For FPN, we assume
# that all networks have 5 spatial reductions, each by a factor of 2. Level 1
# would correspond to the input image, hence it does not make sense to use it.
LOWEST_BACKBONE_LVL = 2   # E.g., "conv2"-like level
HIGHEST_BACKBONE_LVL = 5  # E.g., "conv5"-like level


# ---------------------------------------------------------------------------- #
# FPN with ResNet
# ---------------------------------------------------------------------------- #

def add_fpn_ResNet50_conv5_body(model):
    return add_fpn_onto_conv_body(
        model, ResNet.add_ResNet50_conv5_body, fpn_level_info_ResNet50_conv5
    )


def add_fpn_ResNet50_conv5_P2only_body(model):
    return add_fpn_onto_conv_body(
        model,
        ResNet.add_ResNet50_conv5_body,
        fpn_level_info_ResNet50_conv5,
        P2only=True
    )


def add_fpn_ResNet101_conv5_body(model):
    return add_fpn_onto_conv_body(
        model, ResNet.add_ResNet101_conv5_body, fpn_level_info_ResNet101_conv5
    )


def add_fpn_ResNet101_conv5_P2only_body(model):
    return add_fpn_onto_conv_body(
        model,
        ResNet.add_ResNet101_conv5_body,
        fpn_level_info_ResNet101_conv5,
        P2only=True
    )


def add_fpn_ResNet152_conv5_body(model):
    return add_fpn_onto_conv_body(
        model, ResNet.add_ResNet152_conv5_body, fpn_level_info_ResNet152_conv5
    )


def add_fpn_ResNet152_conv5_P2only_body(model):
    return add_fpn_onto_conv_body(
        model,
        ResNet.add_ResNet152_conv5_body,
        fpn_level_info_ResNet152_conv5,
        P2only=True
    )


# ---------------------------------------------------------------------------- #
# Functions for bolting FPN onto a backbone architectures
# ---------------------------------------------------------------------------- #

def add_fpn_onto_conv_body(
    model, conv_body_func, fpn_level_info_func, P2only=False
):
    """Add the specified conv body to the model and then add FPN levels to it.
    """
    # Note: blobs_conv is in revsersed order: [fpn5, fpn4, fpn3, fpn2]
    # similarly for dims_conv: [2048, 1024, 512, 256]
    # similarly for spatial_scales_fpn: [1/32, 1/16, 1/8, 1/4]

    conv_body_func(model)
    blobs_fpn, dim_fpn, spatial_scales_fpn = add_fpn(
        model, fpn_level_info_func()
    )

    # add pam_fusion
    # need to remove 3*3 conv in FPN,(do conv in pam module)
    if 1:
        # 1 corret blob
        pam_blobs_in = blobs_fpn[::-1]
        pam_blobs_out = add_PAM_fusion(model, pam_blobs_in, dim_fpn, dim_fpn)
        pam_blobs = pam_blobs_out[::-1]


    if P2only:
        # use only the finest level
        return pam_blobs[-1], dim_fpn, spatial_scales_fpn[-1]
    else:
        # use all levels
        return pam_blobs, dim_fpn, spatial_scales_fpn


def add_fpn(model, fpn_level_info):
    """Add FPN connections based on the model described in the FPN paper."""
    # FPN levels are built starting from the highest/coarest level of the
    # backbone (usually "conv5"). First we build down, recursively constructing
    # lower/finer resolution FPN levels. Then we build up, constructing levels
    # that are even higher/coarser than the starting level.
    fpn_dim = cfg.FPN.DIM
    min_level, max_level = get_min_max_levels()
    # Count the number of backbone stages that we will generate FPN levels for
    # starting from the coarest backbone stage (usually the "conv5"-like level)
    # E.g., if the backbone level info defines stages 4 stages: "conv5",
    # "conv4", ... "conv2" and min_level=2, then we end up with 4 - (2 - 2) = 4
    # backbone stages to add FPN to.
    num_backbone_stages = (
        len(fpn_level_info.blobs) - (min_level - LOWEST_BACKBONE_LVL)
    )

    lateral_input_blobs = fpn_level_info.blobs[:num_backbone_stages]
    output_blobs = [
        'fpn_inner_{}'.format(s)
        for s in fpn_level_info.blobs[:num_backbone_stages]
    ]
    fpn_dim_lateral = fpn_level_info.dims
    xavier_fill = ('XavierFill', {})

    # For the coarsest backbone level: 1x1 conv only seeds recursion
    if cfg.FPN.USE_GN:
        # use GroupNorm
        c = model.ConvGN(
            lateral_input_blobs[0],
            output_blobs[0],  # note: this is a prefix
            dim_in=fpn_dim_lateral[0],
            dim_out=fpn_dim,
            group_gn=get_group_gn(fpn_dim),
            kernel=1,
            pad=0,
            stride=1,
            weight_init=xavier_fill,
            bias_init=const_fill(0.0)
        )
        output_blobs[0] = c  # rename it
    else:
        model.Conv(
            lateral_input_blobs[0],
            output_blobs[0],
            dim_in=fpn_dim_lateral[0],
            dim_out=fpn_dim,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=xavier_fill,
            bias_init=const_fill(0.0)
        )

    #
    # Step 1: recursively build down starting from the coarsest backbone level
    #

    # For other levels add top-down and lateral connections
    for i in range(num_backbone_stages - 1):
        add_topdown_lateral_module(
            model,
            output_blobs[i],             # top-down blob
            lateral_input_blobs[i + 1],  # lateral blob
            output_blobs[i + 1],         # next output blob
            fpn_dim,                     # output dimension
            fpn_dim_lateral[i + 1]       # lateral input dimension
        )

    # Post-hoc scale-specific 3x3 convs
    blobs_fpn = []
    spatial_scales = []
    for i in range(num_backbone_stages): 
        blobs_fpn += [output_blobs[i]]
        spatial_scales += [fpn_level_info.spatial_scales[i]]

    #
    # Step 2: build up starting from the coarsest backbone level
    #

    # Check if we need the P6 feature map
    if not cfg.FPN.EXTRA_CONV_LEVELS and max_level == HIGHEST_BACKBONE_LVL + 1:
        # Original FPN P6 level implementation from our CVPR'17 FPN paper
        P6_blob_in = blobs_fpn[0]
        P6_name = P6_blob_in + '_subsampled_2x'
        # Use max pooling to simulate stride 2 subsampling
        P6_blob = model.MaxPool(P6_blob_in, P6_name, kernel=1, pad=0, stride=2)
        blobs_fpn.insert(0, P6_blob)
        spatial_scales.insert(0, spatial_scales[0] * 0.5)

    # Coarser FPN levels introduced for RetinaNet
    if cfg.FPN.EXTRA_CONV_LEVELS and max_level > HIGHEST_BACKBONE_LVL:
        fpn_blob = fpn_level_info.blobs[0]
        dim_in = fpn_level_info.dims[0]
        for i in range(HIGHEST_BACKBONE_LVL + 1, max_level + 1):
            fpn_blob_in = fpn_blob
            if i > HIGHEST_BACKBONE_LVL + 1:
                fpn_blob_in = model.Relu(fpn_blob, fpn_blob + '_relu')
            fpn_blob = model.Conv(
                fpn_blob_in,
                'fpn_' + str(i),
                dim_in=dim_in,
                dim_out=fpn_dim,
                kernel=3,
                pad=1,
                stride=2,
                weight_init=xavier_fill,
                bias_init=const_fill(0.0)
            )
            dim_in = fpn_dim
            blobs_fpn.insert(0, fpn_blob)
            spatial_scales.insert(0, spatial_scales[0] * 0.5)

    return blobs_fpn, fpn_dim, spatial_scales


def add_topdown_lateral_module(
    model, fpn_top, fpn_lateral, fpn_bottom, dim_top, dim_lateral
):
    """Add a top-down lateral module."""
    # Lateral 1x1 conv
    if cfg.FPN.USE_GN:
        # use GroupNorm
        lat = model.ConvGN(
            fpn_lateral,
            fpn_bottom + '_lateral',
            dim_in=dim_lateral,
            dim_out=dim_top,
            group_gn=get_group_gn(dim_top),
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(
                const_fill(0.0) if cfg.FPN.ZERO_INIT_LATERAL
                else ('XavierFill', {})),
            bias_init=const_fill(0.0)
        )
    else:
        lat = model.Conv(
            fpn_lateral,
            fpn_bottom + '_lateral',
            dim_in=dim_lateral,
            dim_out=dim_top,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(
                const_fill(0.0)
                if cfg.FPN.ZERO_INIT_LATERAL else ('XavierFill', {})
            ),
            bias_init=const_fill(0.0)
        )
    # Top-down 2x upsampling
    td = model.net.UpsampleNearest(fpn_top, fpn_bottom + '_topdown', scale=2)
    # Sum lateral and top-down
    model.net.Sum([lat, td], fpn_bottom)


def get_min_max_levels():
    """The min and max FPN levels required for supporting RPN and/or RoI
    transform operations on multiple FPN levels.
    """
    min_level = LOWEST_BACKBONE_LVL
    max_level = HIGHEST_BACKBONE_LVL
    if cfg.FPN.MULTILEVEL_RPN and not cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.RPN_MAX_LEVEL
        min_level = cfg.FPN.RPN_MIN_LEVEL
    if not cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.ROI_MAX_LEVEL
        min_level = cfg.FPN.ROI_MIN_LEVEL
    if cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = max(cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.ROI_MAX_LEVEL)
        min_level = min(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.ROI_MIN_LEVEL)
    return min_level, max_level


# ---------------------------------------------------------------------------- #
# RPN with an FPN backbone
# ---------------------------------------------------------------------------- #

def add_fpn_rpn_outputs(model, blobs_in, dim_in, spatial_scales):
    """Add RPN on FPN specific outputs."""
    num_anchors = len(cfg.FPN.RPN_ASPECT_RATIOS)
    dim_out = dim_in

    k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
    k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid

    #up sampling blobs_in to get large feature map
    # using conv_T

    # use_up sampling rpn or not
    use_fine_anchor = cfg.RPN.FINEANCHOR

    if use_fine_anchor:
        blobs_in_up = []
        for lvl in range(k_min, k_max+1):

            slvl = str(lvl)
            bl_in = blobs_in[::-1][lvl-k_min]

            bl_in_up = model.ConvTranspose(
                bl_in,
                'bl_in_up_' + slvl,
                dim_in,
                dim_out,
                kernel=2,
                pad=0,
                stride=2,
                weight_init=gauss_fill(0.01),
                bias_init=const_fill(0.0)
            )
            bl_in_up = model.Relu(bl_in_up, bl_in_up)

            blobs_in_up += [bl_in_up]

    # assert len(blobs_in) == k_max - k_min + 1

    for lvl in range(k_min, k_max + 1):

        # bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
        # sc = spatial_scales[k_max - lvl]  # in reversed order
        
        if use_fine_anchor:
            bl_in = blobs_in_up[lvl-k_min]
            sc = spatial_scales[::-1][lvl-k_min]
        else:
            bl_in = blobs_in[::-1][lvl-k_min]
            sc = spatial_scales[::-1][lvl-k_min]

        slvl = str(lvl)
        if lvl == k_min:
            # Create conv ops with randomly initialized weights and
            # zeroed biases for the first FPN level; these will be shared by
            # all other FPN levels
            # RPN hidden representation
            conv_rpn_fpn = model.Conv(
                bl_in,
                'conv_rpn_fpn' + slvl,
                dim_in,
                dim_out,
                kernel=3,
                pad=1,
                stride=1,
                weight_init=gauss_fill(0.01),
                bias_init=const_fill(0.0)
            )
            model.Relu(conv_rpn_fpn, conv_rpn_fpn)
            # Proposal classification scores
            adarpn_cls_logits_fpn = model.Conv(
                conv_rpn_fpn,
                'adarpn_cls_logits_fpn' + slvl,
                dim_in,
                1,
                kernel=1,
                pad=0,
                stride=1,
                weight_init=gauss_fill(0.01),
                bias_init=const_fill(0.0)
            )
            # Propasal w and h 
            adarpn_bbox_wh_pred_fpn = model.Conv(
                conv_rpn_fpn,
                'adarpn_bbox_wh_pred_fpn' + slvl,
                dim_in,
                2,
                kernel=1,
                pad=0,
                stride=1,
                weight_init=gauss_fill(0.01),
                bias_init=const_fill(0.0)
            )

            # Proposal bbox regression deltas
            adarpn_bbox_pred_fpn = model.Conv(
                conv_rpn_fpn,
                'adarpn_bbox_pred_fpn' + slvl,
                dim_in,
                4,
                kernel=1,
                pad=0,
                stride=1,
                weight_init=const_fill(0.0),
                bias_init=const_fill(0.0)
            )
        else:
            # Share weights and biases
            sk_min = str(k_min)
            # RPN hidden representation
            conv_rpn_fpn = model.ConvShared(
                bl_in,
                'conv_rpn_fpn' + slvl,
                dim_in,
                dim_out,
                kernel=3,
                pad=1,
                stride=1,
                weight='conv_rpn_fpn' + sk_min + '_w',
                bias='conv_rpn_fpn' + sk_min + '_b'
            )
            model.Relu(conv_rpn_fpn, conv_rpn_fpn)
            # Proposal classification scores
            adarpn_cls_logits_fpn = model.ConvShared(
                conv_rpn_fpn,
                'adarpn_cls_logits_fpn' + slvl,
                dim_in,
                1,
                kernel=1,
                pad=0,
                stride=1,
                weight='adarpn_cls_logits_fpn' + sk_min + '_w',
                bias='adarpn_cls_logits_fpn' + sk_min + '_b'
            )
            adarpn_bbox_wh_pred_fpn = model.ConvShared(
                conv_rpn_fpn,
                'adarpn_bbox_wh_pred_fpn' + slvl,
                dim_in,
                2,
                kernel=1,
                pad=0,
                stride=1,
                weight='adarpn_bbox_wh_pred_fpn' + sk_min + '_w',
                bias='adarpn_bbox_wh_pred_fpn' + sk_min + '_b'
            )
            # Proposal bbox regression deltas
            adarpn_bbox_pred_fpn = model.ConvShared(
                conv_rpn_fpn,
                'adarpn_bbox_pred_fpn' + slvl,
                dim_in,
                4,
                kernel=1,
                pad=0,
                stride=1,
                weight='adarpn_bbox_pred_fpn' + sk_min + '_w',
                bias='adarpn_bbox_pred_fpn' + sk_min + '_b'
            )

        if not model.train or cfg.MODEL.FASTER_RCNN:
            # Proposals are needed during:
            #  1) inference (== not model.train) for RPN only and Faster R-CNN
            #  OR
            #  2) training for Faster R-CNN
            # Otherwise (== training for RPN only), proposals are not needed
            lvl_anchors = generate_anchors(
                stride=2.**lvl,
                sizes=(cfg.FPN.RPN_ANCHOR_START_SIZE * 2.**(lvl - k_min), ),
                aspect_ratios=cfg.FPN.RPN_ASPECT_RATIOS
            )
            adarpn_cls_probs_fpn = model.net.Sigmoid(
                adarpn_cls_logits_fpn, 'adarpn_cls_probs_fpn' + slvl
            )
            model.GenerateProposals(
                [adarpn_cls_probs_fpn, adarpn_bbox_wh_pred_fpn, adarpn_bbox_pred_fpn, 'im_info'],
                ['rpn_rois_fpn' + slvl, 'rpn_roi_probs_fpn' + slvl],
                anchors=lvl_anchors,
                spatial_scale=sc,
                ap_size=cfg.FPN.RPN_ANCHOR_START_SIZE * 2.**(lvl - k_min)
            )


def add_fpn_rpn_attention(model, blobs_in, dim_in, spatial_scales):

    dim_out = dim_in
    k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
    k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid

    pam_blobs_in = blobs_in[::-1]
    pam_blobs = add_PAM_fusion(model, pam_blobs_in, dim_in, dim_out)
    

    for lvl in range(k_min, k_max + 1):

        # bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
        # sc = spatial_scales[k_max - lvl]  # in reversed order

        pam_blob = pam_blobs[lvl-k_min]
        sc = spatial_scales[::-1][lvl-k_min]
        slvl = str(lvl)

        if lvl == k_min:

            # pam_out = add_PAM(model, conv_rpn_fpn, dim_in, dim_out, slvl)
            # Proposal classification scores
            adarpn_cls_logits_fpn = model.Conv(
                pam_blob,
                'adarpn_cls_logits_fpn' + slvl,
                dim_in,
                1,
                kernel=1,
                pad=0,
                stride=1,
                weight_init=gauss_fill(0.01),
                bias_init=const_fill(0.0)
            )
            # Propasal w and h 
            adarpn_bbox_wh_pred_fpn = model.Conv(
                pam_blob,
                'adarpn_bbox_wh_pred_fpn' + slvl,
                dim_in,
                2,
                kernel=1,
                pad=0,
                stride=1,
                weight_init=gauss_fill(0.01),
                bias_init=const_fill(0.0)
            )

            # Proposal bbox regression deltas
            adarpn_bbox_pred_fpn = model.Conv(
                pam_blob,
                'adarpn_bbox_pred_fpn' + slvl,
                dim_in,
                4,
                kernel=1,
                pad=0,
                stride=1,
                weight_init=const_fill(0.0),
                bias_init=const_fill(0.0)
            )
        else:
            # Share weights and biases
            sk_min = str(k_min)

            adarpn_cls_logits_fpn = model.ConvShared(
                pam_blob,
                'adarpn_cls_logits_fpn' + slvl,
                dim_in,
                1,
                kernel=1,
                pad=0,
                stride=1,
                weight='adarpn_cls_logits_fpn' + sk_min + '_w',
                bias='adarpn_cls_logits_fpn' + sk_min + '_b'
            )
            adarpn_bbox_wh_pred_fpn = model.ConvShared(
                pam_blob,
                'adarpn_bbox_wh_pred_fpn' + slvl,
                dim_in,
                2,
                kernel=1,
                pad=0,
                stride=1,
                weight='adarpn_bbox_wh_pred_fpn' + sk_min + '_w',
                bias='adarpn_bbox_wh_pred_fpn' + sk_min + '_b'
            )
            # Proposal bbox regression deltas
            adarpn_bbox_pred_fpn = model.ConvShared(
                pam_blob,
                'adarpn_bbox_pred_fpn' + slvl,
                dim_in,
                4,
                kernel=1,
                pad=0,
                stride=1,
                weight='adarpn_bbox_pred_fpn' + sk_min + '_w',
                bias='adarpn_bbox_pred_fpn' + sk_min + '_b'
            )

        if not model.train or cfg.MODEL.FASTER_RCNN:
            # Proposals are needed during:
            #  1) inference (== not model.train) for RPN only and Faster R-CNN
            #  OR
            #  2) training for Faster R-CNN
            # Otherwise (== training for RPN only), proposals are not needed
            lvl_anchors = generate_anchors(
                stride=2.**lvl,
                sizes=(cfg.FPN.RPN_ANCHOR_START_SIZE * 2.**(lvl - k_min), ),
                aspect_ratios=cfg.FPN.RPN_ASPECT_RATIOS
            )
            adarpn_cls_probs_fpn = model.net.Sigmoid(
                adarpn_cls_logits_fpn, 'adarpn_cls_probs_fpn' + slvl
            )
            model.GenerateProposals(
                [adarpn_cls_probs_fpn, adarpn_bbox_wh_pred_fpn, adarpn_bbox_pred_fpn, 'im_info'],
                ['rpn_rois_fpn' + slvl, 'rpn_roi_probs_fpn' + slvl],
                anchors=lvl_anchors,
                spatial_scale=sc,
                ap_size=cfg.FPN.RPN_ANCHOR_START_SIZE * 2.**(lvl - k_min)
            )




def add_PAM_fusion(model, blobs_in, dim_in, dim_out):
    '''
    input
        blobs_in: FPN [p2, p3, p4, p5] ordel
    return
        pam_blobs_out: [p2_pam_out_up2, p4_pam_out_up1, p4_pam_out, p5] order
    '''
    # adjust pam input
    p2_sub = model.MaxPool(
        blobs_in[0], 'pam_p2_sub1', kernel=3, pad=1, stride=2
    )
    pam_in2 = model.Conv(
        p2_sub, 'pam_in2', dim_in, dim_in,
        kernel=3, pad=1, stride=2,
        weight_init=gauss_fill(0.01), bias_init=const_fill(0.0)
    )
    model.Relu(pam_in2, pam_in2)

    pam_in3 = model.Conv(
        blobs_in[1], 'pam_in3', dim_in, dim_in,
        kernel=3, pad=1, stride=2,
        weight_init=gauss_fill(0.01), bias_init=const_fill(0.0)
    )
    model.Relu(pam_in3, pam_in3)

    pam_in4 = model.Conv(
        blobs_in[2], 'pam_in4', dim_in, dim_in,
        kernel=3, pad=1, stride=1,
        weight_init=gauss_fill(0.01), bias_init=const_fill(0.0)
    )
    model.Relu(pam_in4, pam_in4)

    pam_outs=[]
    for lvl in range(2,5):
        slvl = str(lvl)

        # query
        proj_query = model.Conv(
            'pam_in'+slvl, 'fpn_pam_query'+slvl , dim_in, int(dim_out/8), 
            kernel=1, pad=0, stride=1,
            weight_init=gauss_fill(0.01), bias_init=const_fill(0.0)
        )
        proj_query = model.Reshape(
            proj_query, ['fpn_pam_query_reshape'+slvl, 'pam_query_old_shape'+slvl ],
            shape=[cfg.TRAIN.IMS_PER_BATCH, int(dim_out/8), -1]
        )[0]

        # key
        proj_key = model.Conv(
            'pam_in'+slvl, 'fpn_pam_key'+slvl, dim_in, int(dim_out/8), 
            kernel=1, pad=0, stride=1,
            weight_init=gauss_fill(0.01), bias_init=const_fill(0.0)
        )
        proj_key = model.Reshape(
            proj_key, ['fpn_pam_key_reshape'+slvl, 'pam_key_old_shape'+slvl],
            shape=[cfg.TRAIN.IMS_PER_BATCH, int(dim_out/8), -1]
        )[0]

        # attention
        energy = model.BatchMatMul(
            [proj_query, proj_key], 'fpn_pam_energy'+slvl, trans_a=1, trans_b=0
        )
        attention = model.Softmax(
            energy, 'fpn_pam_attention'+slvl, axis=-1
        )

        # add attention
        prj = model.Conv(
            'pam_in'+slvl, 'fpn_pam_prj'+slvl, dim_in, dim_out,
            kernel=1, pad=0, stride=1,
            weight_init=gauss_fill(0.01), bias_init=const_fill(0.0)
        )
        prj_reshape = model.Reshape(
            prj, ['fpn_pam_prj_reshape'+slvl, 'fpn_pam_prj_old_shape'+slvl],
            shape=[cfg.TRAIN.IMS_PER_BATCH, dim_out, -1]
        )[0]

        pam = model.BatchMatMul(
            [prj_reshape, attention], 'fpn_pam_out'+slvl, trans_a=0, trans_b=1
        )
        pam_reshape = model.Reshape(
            [pam, 'fpn_pam_prj_old_shape'+slvl], ['fpn_pam_out_reshape'+slvl, 'fpn_pam_out_old_shape'+slvl]
        )[0]
        pam_out = model.Add(
            ['pam_in'+slvl, pam_reshape], 'fpn_pam_out'+slvl, 
        )

        # # create params gamma
        # from caffe2.python import core
        # from caffe2.python.modeling import initializers
        # from caffe2.python.modeling.parameter_info import ParameterTags
        # WeightInitializer = initializers.update_initializer(
        #         None, None, ("ConstantFill", {})
        #     )
        # pam_gamma = model.create_param(
        #         param_name='fpn_pam_gamma'+slvl,
        #         shape=[1,],
        #         initializer=WeightInitializer,
        #         tags=ParameterTags.WEIGHT
        #     )
        # inputs = ([pam_gamma,])
        # inputs = core._RectifyInputOutput(inputs)
        # for item in inputs:
        #     if not model.net.BlobIsDefined(item):
        #         assert item.Net() != model.net
        #         model.net.AddExternalInput(item)

        # pam_out = model.WeightedSum(
        #     [pam_reshape, 'fpn_pam_gamma'+slvl], 'fpn_pam_out'+slvl, grad_on_w=True
        # )
        pam_outs += [pam_out]

    # adjust to get output
    # p2 convT --> bilinear_up_sampling
    p2_pam_out_up1 = model.ConvTranspose(
        pam_outs[0], 'p2_pam_out_up1', dim_out, dim_out,
        kernel=2, pad=0, stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    model.Relu(p2_pam_out_up1, p2_pam_out_up1)
    p2_pam_out_up2 = model.BilinearInterpolation(
        'p2_pam_out_up1', 'p2_pam_out_up2', dim_out, dim_out, 2
    )

    # p3
    p4_pam_out_up1 = model.ConvTranspose(
        pam_outs[1], 'p4_pam_out_up1', dim_out, dim_out,
        kernel=2, pad=0, stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    model.Relu(p2_pam_out_up2, p2_pam_out_up2)

    # p4
    p4_pam_out = model.Conv(
        pam_outs[2], 'p4_pam_out', dim_in, dim_in,
        kernel=3, pad=1, stride=1,
        weight_init=gauss_fill(0.01), bias_init=const_fill(0.0)
    )
    
    return [p2_pam_out_up2, p4_pam_out_up1, p4_pam_out, blobs_in[3]]




def add_deform_feature(model, conv_blobs_in, dim_in):
    dim_out = dim_in
    k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
    k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid

    # use_up sampling rpn or not
    use_fine_anchor = cfg.RPN.FINEANCHOR

    if use_fine_anchor:
        blobs_in_up = []
        for lvl in range(k_min, k_max+1):

            slvl = str(lvl)
            bl_in = conv_blobs_in[::-1][lvl-k_min]

            bl_in_up = model.ConvTranspose(
                bl_in,
                'bl_in_up_' + slvl,
                dim_in,
                dim_out,
                kernel=2,
                pad=0,
                stride=2,
                weight_init=gauss_fill(0.01),
                bias_init=const_fill(0.0)
            )
            bl_in_up = model.Relu(bl_in_up, bl_in_up)

            blobs_in_up += [bl_in_up]

    blob_out = conv_blobs_in[:]
    for lvl in range(k_min, k_max + 1):

        if use_fine_anchor:
            conv_bl_in = blobs_in_up[lvl-k_min]
        else:
            conv_bl_in = conv_blobs_in[::-1][lvl-k_min]
        slvl = str(lvl)
        rpnwh_bl_in = 'adarpn_bbox_wh_pred_fpn' + slvl
        
        deform_offset = model.Conv(
                rpnwh_bl_in,
                'deform_offset' + slvl,
                2,
                2*3*3,
                kernel=1,
                pad=0,
                stride=1,
                weight_init=gauss_fill(0.01),
                bias_init=const_fill(0.0)
            )
        batchsize = cfg.TRAIN.IMS_PER_BATCH
        # deform_w = model.net.GaussianFill([], ['deform_conv_' + slvl + '_w'], mean=0.0, std=0.01, shape=[dim_in, 64, 3, 3], run_once=1)
        # deform_b = model.net.ConstantFill([], ['deform_conv_' + slvl + '_b'], shape=[dim_in], value=0.0, run_once=1)

        # deform_w = model.param_init_net.XavierFill([], 'deform_conv_' + slvl + '_w', shape=[dim_in, dim_in, 3, 3])
        # deform_b = model.param_init_net.ConstantFill([], 'deform_conv_' + slvl + '_b', shape=[dim_in,])

        from caffe2.python import core
        from caffe2.python.modeling import initializers
        from caffe2.python.modeling.parameter_info import ParameterTags

        # create params
        WeightInitializer = initializers.update_initializer(
                None, None, ("XavierFill", {})
            )
        deform_w = model.create_param(
                param_name='deform_conv_' + slvl + '_w',
                shape=[dim_in, cfg.FPN.DIM, 3, 3],
                initializer=WeightInitializer,
                tags=ParameterTags.WEIGHT
            )
        BiasInitializer = initializers.update_initializer(
                None, None, ("ConstantFill", {})
            )
        
        deform_b = model.create_param(
                param_name='deform_conv_' + slvl + '_b',
                shape=[dim_in, ],
                initializer=BiasInitializer,
                tags=ParameterTags.BIAS
            )
        inputs = ([conv_bl_in, deform_offset, deform_w, deform_b])
        inputs = core._RectifyInputOutput(inputs)
        for item in inputs:
            if not model.net.BlobIsDefined(item):
                assert item.Net() != model.net
                model.net.AddExternalInput(item)

        op = core.CreateOperator(
            "DeformConv",
            [conv_bl_in, deform_offset, deform_w, deform_b],
            'deform_output' + slvl,
            stride=1,
            kernel=3,
            pad=1,
            order='NCHW',
            engine='CUDNN',
        )
        model.net.Proto().op.extend([op])
        
        blob_out[-1 - (lvl-k_min)] = 'deform_output' + slvl
    return blob_out

    
def add_fpn_rpn_losses(model):
    """Add RPN on FPN specific losses."""
    loss_gradients = {}
    for lvl in range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1):
        slvl = str(lvl)
        # Spatially narrow the full-sized RPN label arrays to match the feature map
        # shape
        model.net.SpatialNarrowAs(
            ['adarpn_labels_int32_wide_fpn' + slvl, 'adarpn_cls_logits_fpn' + slvl],
            'adarpn_labels_int32_fpn' + slvl
        )
        model.net.SpatialNarrowAs(
            ['adarpn_bbox_wh_wide_fpn' + slvl, 'adarpn_bbox_wh_pred_fpn' + slvl],
            'adarpn_bbox_wh_fpn' + slvl
        )
        model.net.SpatialNarrowAs(
            ['adarpn_bbox_wh_inside_wide_fpn' + slvl, 'adarpn_bbox_wh_pred_fpn' + slvl],
            'adarpn_bbox_wh_inside_fpn' + slvl
        )
        model.net.SpatialNarrowAs(
            ['adarpn_bbox_wh_outside_wide_fpn' + slvl, 'adarpn_bbox_wh_pred_fpn' + slvl],
            'adarpn_bbox_wh_outside_fpn' + slvl
        )
        for key in ('delta', 'inside_weights', 'outside_weights'):
            model.net.SpatialNarrowAs(
                [
                    'adarpn_bbox_' + key + '_wide_fpn' + slvl,
                    'adarpn_bbox_pred_fpn' + slvl
                ],
                'adarpn_bbox_' + key + '_fpn' + slvl
            )

        """"
        normalize",
        "(int) default 1; if true, divide the loss by the number of targets > "
        "-1.")
        """
        loss_adarpn_cls_fpn = model.net.SigmoidCrossEntropyLoss(
            ['adarpn_cls_logits_fpn' + slvl, 'adarpn_labels_int32_fpn' + slvl],
            'loss_adarpn_cls_fpn' + slvl,
            normalize=0,
            scale=(
                model.GetLossScale() / cfg.TRAIN.RPN_BATCH_SIZE_PER_IM /
                cfg.TRAIN.IMS_PER_BATCH
            )
        )
        
        # """
        # focal loss
        # """
        # loss_adarpn_cls_fpn = model.net.SigmoidFocalLoss(
        #     ['adarpn_cls_logits_fpn' + slvl, 'adarpn_labels_int32_fpn' + slvl, 'fg_num_batch'],
        #     'loss_adarpn_cls_fpn' + slvl,
        #     gamma=cfg.RETINANET.LOSS_GAMMA, #default value 2
        #     alpha=cfg.RETINANET.LOSS_ALPHA, #default value 0.25
        #     scale=model.GetLossScale(),
        #     num_classes=1                   # RPN only have two class
        # )


        loss_adarpn_bbox_wh_fpn = model.net.SmoothL1Loss(
            [
                'adarpn_bbox_wh_pred_fpn' + slvl, 'adarpn_bbox_wh_fpn' + slvl,
                'adarpn_bbox_wh_inside_fpn' + slvl,
                'adarpn_bbox_wh_outside_fpn' + slvl
            ],
            'loss_adarpn_bbox_wh_fpn' + slvl,
            beta=1. / 9.,
            scale=model.GetLossScale(),
        )

        # Normalization by (1) RPN_BATCH_SIZE_PER_IM and (2) IMS_PER_BATCH is
        # handled by (1) setting bbox outside weights and (2) SmoothL1Loss
        # normalizes by IMS_PER_BATCH
        loss_adarpn_bbox_fpn = model.net.SmoothL1Loss(
            [
                'adarpn_bbox_pred_fpn' + slvl, 'adarpn_bbox_delta_fpn' + slvl,
                'adarpn_bbox_inside_weights_fpn' + slvl,
                'adarpn_bbox_outside_weights_fpn' + slvl
            ],
            'loss_adarpn_bbox_fpn' + slvl,
            beta=1. / 9.,
            scale=0.0,
        )
        loss_gradients.update(
            blob_utils.
            get_loss_gradients(model, [loss_adarpn_cls_fpn, loss_adarpn_bbox_wh_fpn])
            # loss_adarpn_bbox_fpn
        )

        model.AddMetrics(['fg_num_batch', 'bg_num_batch'])
        model.AddLosses(['loss_adarpn_cls_fpn' + slvl, 'loss_adarpn_bbox_wh_fpn' + slvl])
        # , 'loss_adarpn_bbox_fpn' + slvl
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Helper functions for working with multilevel FPN RoIs
# ---------------------------------------------------------------------------- #

def map_rois_to_fpn_levels(rois, k_min, k_max):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """
    # Compute level ids
    s = np.sqrt(box_utils.boxes_area(rois))
    s0 = cfg.FPN.ROI_CANONICAL_SCALE  # default: 224
    lvl0 = cfg.FPN.ROI_CANONICAL_LEVEL  # default: 4

    # Eqn.(1) in FPN paper
    target_lvls = np.floor(lvl0 + np.log2(s / s0 + 1e-6))
    target_lvls = np.clip(target_lvls, k_min, k_max)
    return target_lvls


def add_multilevel_roi_blobs(
    blobs, blob_prefix, rois, target_lvls, lvl_min, lvl_max
):
    """Add RoI blobs for multiple FPN levels to the blobs dict.

    blobs: a dict mapping from blob name to numpy ndarray
    blob_prefix: name prefix to use for the FPN blobs
    rois: the source rois as a 2D numpy array of shape (N, 5) where each row is
      an roi and the columns encode (batch_idx, x1, y1, x2, y2)
    target_lvls: numpy array of shape (N, ) indicating which FPN level each roi
      in rois should be assigned to
    lvl_min: the finest (highest resolution) FPN level (e.g., 2)
    lvl_max: the coarest (lowest resolution) FPN level (e.g., 6)
    """
    rois_idx_order = np.empty((0, ))
    rois_stacked = np.zeros((0, 5), dtype=np.float32)  # for assert
    for lvl in range(lvl_min, lvl_max + 1):
        idx_lvl = np.where(target_lvls == lvl)[0]
        blobs[blob_prefix + '_fpn' + str(lvl)] = rois[idx_lvl, :]
        rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
        rois_stacked = np.vstack(
            [rois_stacked, blobs[blob_prefix + '_fpn' + str(lvl)]]
        )
    rois_idx_restore = np.argsort(rois_idx_order).astype(np.int32, copy=False)
    blobs[blob_prefix + '_idx_restore_int32'] = rois_idx_restore
    # Sanity check that restore order is correct
    assert (rois_stacked[rois_idx_restore] == rois).all()


# ---------------------------------------------------------------------------- #
# FPN level info for stages 5, 4, 3, 2 for select models (more can be added)
# ---------------------------------------------------------------------------- #

FpnLevelInfo = collections.namedtuple(
    'FpnLevelInfo',
    ['blobs', 'dims', 'spatial_scales']
)


def fpn_level_info_ResNet50_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_5_sum', 'res3_3_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_ResNet101_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_22_sum', 'res3_3_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_ResNet152_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_35_sum', 'res3_7_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )
