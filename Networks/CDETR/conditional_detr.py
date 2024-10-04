# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import math

import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss, sigmoid_focal_loss2)
from .transformer import build_transformer


class ConditionalDETR(nn.Module):
    """ This is the Conditional DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, channel_point, aux_loss=False, dm_decoder=None, branch_merge=False, branch_merge_way=1, transformer_flag="",decoding_arch=''):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer 
        hidden_dim = transformer.d_model #d_model=256
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.point_embed = MLP(hidden_dim, hidden_dim, channel_point, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init point_mebed
        nn.init.constant_(self.point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.point_embed.layers[-1].bias.data, 0)

        # added by gls
        self.dm_decoder=dm_decoder
        self.branch_merge=branch_merge
        self.branch_merge_way=branch_merge_way
        self.transformer_flag=transformer_flag
        self.decoding_arch=decoding_arch

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_points": The normalized points coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # print(pos[-1].shape)
        # pos[-1]=F.interpolate(pos[-1],scale_factor=2)
        shallow_feature = features[0].decompose()[0]  # 浅层特征图
        medium_feature = features[1].decompose()[0]  # 中层特征图
        deep_feature = features[2].decompose()[0]  # 深层特征图
        src, mask = features[-1].decompose()
        # 上采样 mask 和 pos
        if self.decoding_arch=='norm' or self.decoding_arch=='multi_resolution_loss' or self.decoding_arch=='feature_weighting':
            mask_upsampled = F.interpolate(mask.unsqueeze(1).float(), size=(32, 32), mode='nearest').squeeze(1).bool()
            # mask_upsampled = F.interpolate(mask.unsqueeze(1).float(), size=(32, 32), mode='nearest', align_corners=False).squeeze(1).bool()
            pos_upsampled = F.interpolate(pos[-1], size=(32, 32), mode='bilinear', align_corners=False)
        # elif self.decoding_arch=='feature_weighting':
        #     mask_upsampled = F.interpolate(mask.unsqueeze(1).float(), size=(64, 64), mode='nearest').squeeze(1).bool()
        #     # mask_upsampled = F.interpolate(mask.unsqueeze(1).float(), size=(32, 32), mode='nearest', align_corners=False).squeeze(1).bool()
        #     pos_upsampled = F.interpolate(pos[-1], size=(64, 64), mode='bilinear', align_corners=False)
        # mask=mask.float()
        # mask=F.interpolate(mask,scale_factor=2)
        if self.dm_decoder is not None:
            out_dm=self.dm_decoder(shallow_feature,medium_feature,deep_feature,src,self.decoding_arch)
            # import pdb; pdb.set_trace()
            if self.branch_merge:
                # if self.branch_merge_way==1:
                #     src=src+src*F.sigmoid(F.interpolate(out_dm[0],scale_factor=0.5))
                if self.branch_merge_way==2:
                    # out_density=F.interpolate(out_dm[0],scale_factor=0.5) #[8,256,16,16]                 
                    # src=self.input_proj(src)+out_density
                    if self.decoding_arch=='norm':
                        out_density=out_dm[0]
                        out_density1=out_dm[3]
                        out_density2=out_dm[6]
                        out_density3=out_dm[9]
                        src1=self.input_proj(src)#2048->256
                        src2=F.interpolate(src1,scale_factor=2)#16X16->32X32
                        # out_density=F.interpolate(out_dm[0],scale_factor=0.5) #[8,256,16,16] 
                        # out_density1=F.interpolate(out_dm[3],scale_factor=0.5) #[8,256,16,16] 
                        # out_density2=F.interpolate(out_dm[6],scale_factor=0.5) #[8,256,16,16] 
                        # out_density3=F.interpolate(out_dm[9],scale_factor=0.5) #[8,256,16,16] 

                        src=src2+out_density+out_density1+out_density2+out_density3#[8,256,32,32]
                    elif self.decoding_arch=='multi_resolution_loss' or self.decoding_arch=='feature_weighting':
                        out_density=out_dm[0]
                        src=out_dm[0]#[8,256,32,32]

        else:
            out_dm=None

        assert mask is not None
        if self.dm_decoder is not None:
            if self.branch_merge and self.branch_merge_way==2:
                if self.transformer_flag=="merge" or self.transformer_flag=="merge2" or self.transformer_flag=="merge3":
                    hs, reference = self.transformer(src, mask_upsampled, self.query_embed.weight, pos_upsampled, out_density)
                # else:
                #     hs, reference = self.transformer(src, mask, self.query_embed.weight, pos[-1])
            elif not self.branch_merge:
                hs, reference = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])
        else:
            hs, reference = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])


        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            tmp = self.point_embed(hs[lvl])
            tmp[..., :2] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)

        outputs_class = self.class_embed(hs)
        out = {'pred_logits': outputs_class[-1], 'pred_points': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        if out_dm is None:
            return out
        else:
            return [out,out_dm]

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_points': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class dm_decoder2(nn.Module):
    def __init__(self,dim_feedforward=2048, hidden_dim=256, hidden_dim2=128,hidden_dim3=512,hidden_dim4=1024,hidden_dim5=3840):
        super(dm_decoder2, self).__init__()
        self.reg_layer=nn.Sequential(
            nn.Conv2d(dim_feedforward, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            )
        self.reglayer1=nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim2,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
        self.reglayer2=nn.Sequential(
            nn.Conv2d(hidden_dim3,hidden_dim,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # self.reglayer3=nn.Sequential(
        #     nn.Conv2d(hidden_dim4, hidden_dim, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(hidden_dim, hidden_dim2, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        # )
        self.reglayer3=nn.Sequential(
            nn.Conv2d(hidden_dim4, hidden_dim3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim3, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Conv2d(128, 1, 1)
        self.density_layer1=nn.Conv2d(256,1,1)
        self.reg_layer2=nn.Sequential(
            nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim2, hidden_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            )
        self.layer1=nn.Sequential(
            nn.Conv2d(dim_feedforward,dim_feedforward,kernel_size=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_feedforward,hidden_dim4,kernel_size=1,padding=0),
            nn.ReLU(inplace=True),
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(dim_feedforward, hidden_dim4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim4, hidden_dim3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(hidden_dim4, hidden_dim3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim3, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer4=nn.Sequential(
            nn.Conv2d(hidden_dim5,dim_feedforward, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_feedforward, hidden_dim4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim4,hidden_dim3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim3, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer5=nn.Sequential(
            nn.Conv2d(hidden_dim3, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer6=nn.Sequential(
            nn.Conv2d(dim_feedforward, hidden_dim4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim4, hidden_dim3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim3, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer7=self.layer5=nn.Sequential(
            nn.Conv2d(hidden_dim3, hidden_dim3, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim3, hidden_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.conv_weight=nn.Conv2d(hidden_dim4,4,kernel_size=1,padding=0)
        # self.dm_activation = nn.ReLU()
    def forward(self,shallow_feature,medium_feature,deep_feature,x,decoding_arch):
        if decoding_arch=='norm':
            x = F.upsample_bilinear(x, scale_factor=2) #[8,2048,32,32]
            x = self.reg_layer(x) #[8,128,32,32]
            x2=x.clone() #[8,128,32,32]
            mu = self.density_layer(x) #[8,1,32,32]
            mu2 = F.relu(mu) #[8,1,32,32]
            B, C, H, W=mu2.size() 
            mu2_sum = mu2.view([B,-1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3) #[8,1,1,1]
            mu2_normed = mu2 / (mu2_sum + 1e-6) #[8,1,32,32]

            shallow_feature = F.upsample_bilinear(shallow_feature, scale_factor=0.5)#[8,256,32,32]
            shallow=self.reglayer1(shallow_feature) #[8,128,32,32]
            x_shallow=shallow.clone()#[8,128,32,32]
            mu_shallow = self.density_layer(shallow) #[8,1,32,32]
            mu2_shallow = F.relu(mu_shallow) #[8,1,32,32]
            B, C, H, W=mu2_shallow.size() 
            mu2_sum_shallow = mu2_shallow.view([B,-1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3) #[8,1,1,1]
            mu2_normed_shallow = mu2_shallow / (mu2_sum_shallow + 1e-6) #[8,1,32,32]

            medium=self.reglayer2(medium_feature) #[8,128,32,32]
            x_medium=medium.clone()#[8,128,32,32]
            mu_medium = self.density_layer(medium) #[8,1,32,32]
            mu2_medium = F.relu(mu_medium) #[8,1,32,32]
            B, C, H, W=mu2_medium.size() 
            mu2_sum_medium = mu2_medium.view([B,-1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3) #[8,1,1,1]
            mu2_normed_medium = mu2_medium / (mu2_sum_medium + 1e-6) #[8,1,32,32]

            deep_feature = F.upsample_bilinear(deep_feature, scale_factor=2)#[8,1024,32,32]
            deep=self.reglayer3(deep_feature) #[8,128,32,32]
            x_deep=deep.clone()
            mu_deep = self.density_layer(deep) #[8,1,32,32]
            mu2_deep = F.relu(mu_deep) #[8,1,32,32]
            B, C, H, W=mu2_deep.size() 
            mu2_sum_deep = mu2_deep.view([B,-1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3) #[8,1,1,1]
            mu2_normed_deep = mu2_deep / (mu2_sum_deep + 1e-6) #[8,1,32,32]
            return [self.reg_layer2(x2), mu2, mu2_normed,self.reg_layer2(x_shallow),mu2_shallow,mu2_normed_shallow,self.reg_layer2(x_medium),mu2_medium,mu2_normed_medium,self.reg_layer2(x_deep),mu2_deep,mu2_normed_deep]

        elif decoding_arch =='multi_resolution_loss':
            x=self.layer1(x)#[8,1024,16,16]
            ft=torch.cat((x,deep_feature),1)#[8,1024+1024,16,16]
            ft=F.upsample_bilinear(ft,scale_factor=2) #[8,2048,32,32]
            ft=self.layer2(ft) #[8,512,32,32]
            ft=torch.cat((ft,medium_feature),1)#[8,512+512,32,32]
            ft=F.upsample_bilinear(ft,scale_factor=2) #[8,1024,64,64]
            ft=self.layer3(ft) #[8,256,64,64]
            ft=torch.cat((ft,shallow_feature),1) #[8,256+256,64,64]
            ft=F.upsample_bilinear(ft,scale_factor=0.5) #[8,512,32,32]
            ft=self.layer5(ft) #[8,128,32,32]
            ft_clone=ft.clone()
            ft1=self.density_layer(ft)
            ft2=F.relu(ft1)
            B,C,H,W=ft2.size()
            ft2_sum = ft2.view([B,-1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3) #[8,1,1,1]
            ft2_normed = ft2 / (ft2_sum + 1e-6) #[8,1,32,32]
            return self.reg_layer2(ft_clone),ft2,ft2_normed
         
        elif decoding_arch=='feature_weighting':
            x_up=F.upsample_bilinear(x,scale_factor=4) #[8,2048,64,64]
            x_up=self.layer6(x_up) #[8,256,64,64]
            deep_feature_up=F.upsample_bilinear(deep_feature,scale_factor=4) #[8,1024,64,64]
            deep_feature_up=self.layer3(deep_feature_up) #[8,256,64,64]
            medium_feature_up=F.upsample_bilinear(medium_feature,scale_factor=2) #[8,512,64,64]
            medium_feature_up=self.layer7(medium_feature_up) #[8,256,64,64]
            ft_weighting=torch.cat((x_up,deep_feature_up,medium_feature_up,shallow_feature),1) #[8,256+256+256+256,64,64]
            weighting=self.conv_weight(ft_weighting)
            # weighting=F.conv2d(1024,channel=4,activation='softmax') #8 4X64X64
            weighting = F.softmax(weighting, dim=1)
            # 使用权重进行加权
            ft = x_up * weighting[:, 0:1] + deep_feature_up * weighting[:, 1:2] + medium_feature_up * weighting[:, 2:3] + shallow_feature * weighting[:, 3:4]
            ft=F.upsample_bilinear(ft,scale_factor=0.5) #[8,256,32,32]
            ft=self.reglayer1(ft)#[8,128,32,32]
            ft_clone=ft.clone()
            ft1=self.density_layer(ft)
            ft2=F.relu(ft1)
            B,C,H,W=ft2.size()
            ft2_sum = ft2.view([B,-1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3) #[8,1,1,1]
            ft2_normed = ft2 / (ft2_sum + 1e-6) #[8,1,32,32]
            return self.reg_layer2(ft_clone),ft2,ft2_normed
        
# class dm_decoder2(nn.Module):
#     def __init__(self,dim_feedforward=2048, hidden_dim=256, hidden_dim2=128,hidden_dim3=512,hidden_dim4=1024):
#         super(dm_decoder2, self).__init__()
#         self.reg_layer=nn.Sequential(
#             nn.Conv2d(dim_feedforward, hidden_dim, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim, hidden_dim2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             )
#         self.reglayer1=nn.Sequential(
#             nn.Conv2d(hidden_dim,hidden_dim2,kernel_size=3,padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.reglayer2=nn.Sequential(
#             nn.Conv2d(hidden_dim3,hidden_dim,kernel_size=3,padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim, hidden_dim2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.reglayer3=nn.Sequential(
#             nn.Conv2d(hidden_dim4, hidden_dim, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim, hidden_dim2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.density_layer1=nn.Conv2d(hidden_dim, hidden_dim2, 1)
#         self.density_layer2 = nn.Conv2d(hidden_dim2, hidden_dim2, 1)
#         self.density_layer = nn.Conv2d(128, 1, 1)
#         self.reg_layer2=nn.Sequential(
#             nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim2, hidden_dim, kernel_size=1, padding=0),
#             nn.ReLU(inplace=True),
#             )
#         # self.dm_activation = nn.ReLU()
#     def forward(self,shallow_feature,medium_feature,x):
#         x = F.upsample_bilinear(x, scale_factor=2) #[8,2048,32,32]
#         x = self.reg_layer(x) #[8,128,32,32]
#         x2=x.clone() #[8,128,32,32]
#         mu = self.density_layer(x) #[8,1,32,32]
#         mu2 = F.relu(mu) #[8,1,32,32]
#         B, C, H, W=mu2.size() 
#         mu2_sum = mu2.view([B,-1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3) #[8,1,1,1]
#         mu2_normed = mu2 / (mu2_sum + 1e-6) #[8,1,32,32]

#         shallow_feature = F.upsample_bilinear(shallow_feature, scale_factor=0.5)#[8,256,32,32]
#         shallow=self.reglayer1(shallow_feature) #[8,128,32,32]
#         x_shallow=shallow.clone()#[8,128,32,32]
#         mu_shallow = self.density_layer(x_shallow) #[8,1,32,32]
#         mu2_shallow = F.relu(mu_shallow) #[8,1,32,32]
#         B, C, H, W=mu2_shallow.size() 
#         mu2_sum_shallow = mu2_shallow.view([B,-1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3) #[8,1,1,1]
#         mu2_normed_shallow = mu2_shallow / (mu2_sum_shallow + 1e-6) #[8,1,32,32]

#         # medium_feature = F.upsample_bilinear(medium_feature, scale_factor=2)#[8,512,64,64]
#         medium=self.reglayer2(medium_feature) #[8,128,32,32]
#         x_medium=medium.clone()#[8,128,32,32]
#         mu_medium = self.density_layer(x_medium) #[8,1,32,32]
#         mu2_medium = F.relu(mu_medium) #[8,1,32,32]
#         B, C, H, W=mu2_medium.size() 
#         mu2_sum_medium = mu2_medium.view([B,-1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3) #[8,1,1,1]
#         mu2_normed_medium = mu2_medium / (mu2_sum_medium + 1e-6) #[8,1,32,32]

#         # deep_feature = F.upsample_bilinear(deep_feature, scale_factor=2)#[8,1024,32,32]
#         # deep=self.reglayer3(deep_feature) #[8,128,32,32]
#         # x_deep=deep.clone()
#         # mu_deep = self.density_layer(x_deep) #[8,1,32,32]
#         # mu2_deep = F.relu(mu_deep) #[8,1,32,32]
#         # B, C, H, W=mu2_deep.size() 
#         # mu2_sum_deep = mu2_deep.view([B,-1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3) #[8,1,1,1]
#         # mu2_normed_deep = mu2_deep / (mu2_sum_deep + 1e-6) #[8,1,32,32]

#         return [self.reg_layer2(x2), mu2, mu2_normed,self.reg_layer2(x_shallow),mu2_shallow,mu2_normed_shallow,self.reg_layer2(x_medium),mu2_medium,mu2_normed_medium]


# class dm_decoder2(nn.Module):
#     def __init__(self,dim_feedforward=2048, hidden_dim=256, hidden_dim2=128):
#         super(dm_decoder2, self).__init__()
#         self.reg_layer=nn.Sequential(
#             nn.Conv2d(dim_feedforward, hidden_dim, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim, hidden_dim2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             )
#         self.density_layer = nn.Conv2d(128, 1, 1)
#         self.reg_layer2=nn.Sequential(
#             nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim2, hidden_dim, kernel_size=1, padding=0),
#             nn.ReLU(inplace=True),
#             )
#         # self.dm_activation = nn.ReLU()
#     def forward(self,x):
#         x = F.upsample_bilinear(x, scale_factor=2)
#         x = self.reg_layer(x)
#         x2=x.clone()
#         mu = self.density_layer(x)
#         mu2 = F.relu(mu)
#         B, C, H, W=mu2.size()
#         mu2_sum = mu2.view([B,-1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
#         mu2_normed = mu2 / (mu2_sum + 1e-6)
#         return [self.reg_layer2(x2), mu2, mu2_normed]

class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth points and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, with_weights=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.with_weights=with_weights

    def loss_labels(self, outputs, targets, indices, num_points, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_points]
        """
        # import pdb; pdb.set_trace()
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']    # bs, num_query, 2

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).cuda()
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        if not self.with_weights:
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_points, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        # else:
        #     weights=self.loss_labels_weights(outputs, targets, indices, num_points)
        #     loss_ce = sigmoid_focal_loss2(src_logits, target_classes_onehot, num_points, weights, alpha=self.focal_alpha, gamma=2) * \
        #           src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_points):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty points
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_points(self, outputs, targets, indices, num_points):
        """Compute the losses related to the bounding points, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "points" containing a tensor of dim [nb_target_points, 4]
           The target points are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0).cuda()

        loss_point = F.l1_loss(src_points, target_points, reduction='none')

        losses = {}
        losses['loss_point'] = loss_point.sum() / num_points

        # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        #     box_ops.box_cxcywh_to_xyxy(src_points),
        #     box_ops.box_cxcywh_to_xyxy(target_points)))
        # losses['loss_giou'] = loss_giou.sum() / num_points
        # losses['loss_giou'] = 0.0
        return losses

    def loss_masks(self, outputs, targets, indices, num_points):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_points, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_points),
            "loss_dice": dice_loss(src_masks, target_masks, num_points),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'points': self.loss_points,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target points accross all nodes, for normalization purposes
        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_points, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_point = outputs['pred_logits'], outputs['pred_points']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_points = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        points = box_ops.box_cxcywh_to_xyxy(out_point)
        points = torch.gather(points, 1, topk_points.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        points = points * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'points': b} for s, l, b in zip(scores, labels, points)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 2 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    if args.dm_count:
        print("adding a small branch to train a density map")
        if args.branch_merge_way==2:
            density_decoder=dm_decoder2()
    else:
        density_decoder=None

    model = ConditionalDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        channel_point = args.channel_point,
        aux_loss=args.aux_loss,
        dm_decoder=density_decoder,
        branch_merge=args.branch_merge,
        branch_merge_way=args.branch_merge_way,
        transformer_flag=args.transformer_flag,
        decoding_arch=args.decoding_arch
    )

    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_point': args.point_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'points', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses, with_weights=args.with_weights)
    criterion.to(device)
    postprocessors = {'point': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
