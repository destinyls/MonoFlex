import torch
import pdb
import numpy as np
import torch
import pdb
import numpy as np
from torch import nn
from torch.nn import functional as F

from utils.registry import Registry
from model import registry
from model.layers.utils import sigmoid_hm
from model.make_layers import group_norm, _fill_fc_weights
from model.layers.utils import select_point_of_interest
from model.backbone.DCNv2.dcn_v2 import DCNv2
from model.head.detector_loss import make_loss_evaluator
from model.layers.utils import (
	nms_hm,
	select_topk,
	select_point_of_interest,
)

from inplace_abn import InPlaceABN

@registry.PREDICTOR.register("Base_Predictor")
class _predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(_predictor, self).__init__()
        # ("Car", "Cyclist", "Pedestrian")
        classes = len(cfg.DATASETS.DETECT_CLASSES)

        self.regression_head_cfg = cfg.MODEL.HEAD.REGRESSION_MERGE_HEADS
        self.regression_channel_cfg = cfg.MODEL.HEAD.REGRESSION_MERGE_CHANNELS
        self.output_width = cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_height = cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.head_conv = cfg.MODEL.HEAD.NUM_CHANNEL

        use_norm = cfg.MODEL.HEAD.USE_NORMALIZATION
        if use_norm == 'BN': norm_func = nn.BatchNorm2d
        elif use_norm == 'GN': norm_func = group_norm
        else: norm_func = nn.Identity

        # the inplace-abn is applied to reduce GPU memory and slightly increase the batch-size
        self.use_inplace_abn = cfg.MODEL.INPLACE_ABN
        self.bn_momentum = cfg.MODEL.HEAD.BN_MOMENTUM
        self.abn_activision = 'leaky_relu'

        self.loss_evaluator = make_loss_evaluator(cfg)
        self.max_detection = cfg.TEST.DETECTIONS_PER_IMG

        ###########################################
        ###############  Cls Heads ################
        ########################################### 
        if self.use_inplace_abn:
            self.class_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                InPlaceABN(self.head_conv, momentum=self.bn_momentum, activation=self.abn_activision),
                nn.Conv2d(self.head_conv, classes, kernel_size=1, padding=1 // 2, bias=True),
            )
        else:
            self.class_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                norm_func(self.head_conv, momentum=self.bn_momentum), nn.ReLU(inplace=True),
                nn.Conv2d(self.head_conv, classes, kernel_size=1, padding=1 // 2, bias=True),
            )
        
        self.class_head[-1].bias.data.fill_(- np.log(1 / cfg.MODEL.HEAD.INIT_P - 1))

        ###########################################
        ############  Regression Heads ############
        ###########################################
        if self.use_inplace_abn:
            self.box2d_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                InPlaceABN(self.head_conv, momentum=self.bn_momentum, activation=self.abn_activision),
                nn.Conv2d(self.head_conv + 384, 4, kernel_size=1, padding=1 // 2, bias=True)
            )
            self.box3d_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                InPlaceABN(self.head_conv, momentum=self.bn_momentum, activation=self.abn_activision),
                nn.Conv2d(self.head_conv + 384, 21, kernel_size=1, padding=1 // 2, bias=True)
            )
            self.corners_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                InPlaceABN(self.head_conv, momentum=self.bn_momentum, activation=self.abn_activision),
                nn.Conv2d(self.head_conv + 384, 23, kernel_size=1, padding=1 // 2, bias=True)
            )
            self.offset3d_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                InPlaceABN(self.head_conv, momentum=self.bn_momentum, activation=self.abn_activision),
                nn.Conv2d(self.head_conv, 2, kernel_size=1, padding=1 // 2, bias=True)
            )
        else:
            self.box2d_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                norm_func(self.head_conv, momentum=self.bn_momentum), nn.ReLU(inplace=True),
                nn.Conv2d(self.head_conv + 384, 4, kernel_size=1, padding=1 // 2, bias=True)
            )
            self.box3d_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                norm_func(self.head_conv, momentum=self.bn_momentum), nn.ReLU(inplace=True),
                nn.Conv2d(self.head_conv + 384, 21, kernel_size=1, padding=1 // 2, bias=True)
            )
            self.corners_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                norm_func(self.head_conv, momentum=self.bn_momentum), nn.ReLU(inplace=True),
                nn.Conv2d(self.head_conv + 384, 23, kernel_size=1, padding=1 // 2, bias=True)
            )
            self.offset3d_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                norm_func(self.head_conv, momentum=self.bn_momentum), nn.ReLU(inplace=True),
                nn.Conv2d(self.head_conv, 2, kernel_size=1, padding=1 // 2, bias=True),
            )
        _fill_fc_weights(self.box2d_head, 0)
        _fill_fc_weights(self.box3d_head, 0)
        _fill_fc_weights(self.corners_head, 0)
        _fill_fc_weights(self.offset3d_head, 0)

        ###########################################
        ##############  Edge Feature ##############
        ###########################################
        # edge feature fusion
        self.enable_edge_fusion = cfg.MODEL.HEAD.ENABLE_EDGE_FUSION
        self.edge_fusion_kernel_size = cfg.MODEL.HEAD.EDGE_FUSION_KERNEL_SIZE

        if self.enable_edge_fusion:
            trunc_norm_func = nn.BatchNorm1d if cfg.MODEL.HEAD.EDGE_FUSION_NORM == 'BN' else nn.Identity
            trunc_activision_func = nn.Identity()
            # trunc_activision_func = nn.ReLU(inplace=True)
            self.trunc_heatmap_conv = nn.Sequential(
                nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size, padding=self.edge_fusion_kernel_size // 2, padding_mode='replicate'),
                trunc_norm_func(self.head_conv, momentum=self.bn_momentum), trunc_activision_func, nn.Conv1d(self.head_conv, classes, kernel_size=1),
            )
            
            self.trunc_offset_conv = nn.Sequential(
                nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size, padding=self.edge_fusion_kernel_size // 2, padding_mode='replicate'),
                trunc_norm_func(self.head_conv, momentum=self.bn_momentum), trunc_activision_func, nn.Conv1d(self.head_conv, 2, kernel_size=1),
            )

    def forward(self, features, targets):
        up_level16, up_level8, up_level4 = features[0], features[1], features[2]
        b, c, h, w = up_level4.shape

        # output classification
        feature_cls = self.class_head[:-1](up_level4)
        output_cls = self.class_head[-1](feature_cls)
        
        output_regs = []
        feature_offset3d = self.offset3d_head[:-1](up_level4)
        output_offset3d = self.offset3d_head[-1](feature_offset3d)
        if self.enable_edge_fusion:
            edge_indices = torch.stack([t.get_field("edge_indices") for t in targets]) # B x K x 2
            edge_lens = torch.stack([t.get_field("edge_len") for t in targets]) # B 
            # normalize
            grid_edge_indices = edge_indices.view(b, -1, 1, 2).float()
            grid_edge_indices[..., 0] = grid_edge_indices[..., 0] / (self.output_width - 1) * 2 - 1
            grid_edge_indices[..., 1] = grid_edge_indices[..., 1] / (self.output_height - 1) * 2 - 1
            # apply edge fusion for both offset and heatmap
            feature_for_fusion = torch.cat((feature_cls, feature_offset3d), dim=1)
            edge_features = F.grid_sample(feature_for_fusion, grid_edge_indices, align_corners=True).squeeze(-1)
            edge_cls_feature = edge_features[:, :self.head_conv, ...]
            edge_offset_feature = edge_features[:, self.head_conv:, ...]
            edge_cls_output = self.trunc_heatmap_conv(edge_cls_feature)
            edge_offset_output = self.trunc_offset_conv(edge_offset_feature)
            for k in range(b):
                edge_indice_k = edge_indices[k, :edge_lens[k]]
                output_cls[k, :, edge_indice_k[:, 1], edge_indice_k[:, 0]] += edge_cls_output[k, :, :edge_lens[k]]
                output_offset3d[k, :, edge_indice_k[:, 1], edge_indice_k[:, 0]] += edge_offset_output[k, :, :edge_lens[k]]

        feature_box2d = self.box2d_head[:-1](up_level4)
        feature_box3d = self.box3d_head[:-1](up_level4)
        feature_corners = self.corners_head[:-1](up_level4)

        output_cls = sigmoid_hm(output_cls)
        if self.training:
            targets_heatmap, targets_variables = self.loss_evaluator.prepare_targets(targets)
            proj_points = targets_variables["target_centers"]
        else:
            # select top-k of the predicted heatmap
            heatmap = nms_hm(output_cls)
            scores, indexs, clses, ys, xs = select_topk(heatmap, K=self.max_detection)
            proj_points = torch.cat([xs.view(-1, 1), ys.view(-1, 1)], dim=1).unsqueeze(0)
        
        proj_points_8 = proj_points // 2
        proj_points_16 = proj_points // 4

        # 1/8 [N, K, 128]
        up_level8_pois = select_point_of_interest(b, proj_points_8, up_level8)    
        # 1/16 [N, K, 256]
        up_level16_pois = select_point_of_interest(b, proj_points_16, up_level16)

        feature_box2d_pois = select_point_of_interest(b, proj_points, feature_box2d)
        feature_box3d_pois = select_point_of_interest(b, proj_points, feature_box3d)
        feature_corners_pois = select_point_of_interest(b, proj_points, feature_corners)
        
        # [N, K. 640]
        feature_box2d_pois = torch.cat((feature_box2d_pois, up_level8_pois, up_level16_pois), dim=-1)
        # [N, K, 640]
        feature_box3d_pois = torch.cat((feature_box3d_pois, up_level8_pois, up_level16_pois), dim=-1)
        # [N, K, 640]
        feature_corners_pois = torch.cat((feature_corners_pois, up_level8_pois, up_level16_pois), dim=-1)
        
        # [N, 640, K, 1]
        feature_box2d_pois = feature_box2d_pois.permute(0, 2, 1).contiguous().unsqueeze(-1)
        # [N, 640, K, 1]
        feature_box3d_pois = feature_box3d_pois.permute(0, 2, 1).contiguous().unsqueeze(-1)
        # [N, 640, K, 1]
        feature_corners_pois = feature_corners_pois.permute(0, 2, 1).contiguous().unsqueeze(-1)
        
        # [N, C, K, 1]
        output_box2d = self.box2d_head[-1](feature_box2d_pois)
        # [N, C, K, 1]
        output_box3d = self.box3d_head[-1](feature_box3d_pois)
        # [N, C, K, 1]
        output_corners = self.corners_head[-1](feature_corners_pois)

        # [N, K, C]
        output_box2d = output_box2d.squeeze(-1).permute(0, 2, 1).contiguous()
        # [N, K, C]
        output_box3d = output_box3d.squeeze(-1).permute(0, 2, 1).contiguous()
        # [N, K, C]
        output_corners = output_corners.squeeze(-1).permute(0, 2, 1).contiguous()

        # [N, K, 2]
        output_offset3d_pois = select_point_of_interest(b, proj_points, output_offset3d)

        output_regs.append(output_box2d)
        output_regs.append(output_offset3d_pois)
        output_regs.append(output_corners)
        output_regs.append(output_box3d)

        # [N, K, 50]
        output_regs = torch.cat((output_box2d, output_offset3d_pois, output_corners, output_box3d), dim=-1)
        if self.training:
            return [output_cls,  output_regs, targets_heatmap, targets_variables]
        else:
            return {'cls': output_cls, 'reg': output_regs, 'scores': scores, 'clses': clses, 'proj_points': proj_points}

def make_predictor(cfg, in_channels):
    func = registry.PREDICTOR[cfg.MODEL.HEAD.PREDICTOR]
    return func(cfg, in_channels)

