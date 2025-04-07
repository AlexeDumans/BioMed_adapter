import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image



# Residual CLIP Adapter
class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y

        
class CLIP_Inplanted(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = features
        self.seg_adapters = nn.ModuleList( [ClipAdapter(768, bottleneck=512) for i in range(len(features))] )
        self.det_adapters = nn.ModuleList( [ClipAdapter(768, bottleneck=512) for i in range(len(features))] )


    def forward(self, x):
        x = self.image_encoder.trunk.patch_embed(x)
        
        cls_token = self.image_encoder.trunk.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        x = x + self.image_encoder.trunk.pos_embed

        seg_patch_tokens = []
        det_patch_tokens = []

        for i, block in enumerate(self.image_encoder.trunk.blocks):
            x = block(x)
            if (i + 1) in self.features:
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i+1)](x)
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i+1)](x)
                
                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out
                
                if i == 0:
                    seg_patch_tokens = [seg_adapt_med]
                    det_patch_tokens = [det_adapt_med]
                else:
                    seg_patch_tokens.append(seg_adapt_med)
                    det_patch_tokens.append(det_adapt_med)

        x = self.image_encoder.trunk.norm(x)
        
        image_features = x[:, 0]
        
        if hasattr(self.image_encoder, 'head'):
            image_features = self.image_encoder.head.proj(image_features)

        return image_features, seg_patch_tokens, det_patch_tokens




