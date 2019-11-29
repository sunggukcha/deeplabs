import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.backbone import build_backbone

def gn(planes):
    return nn.GroupNorm(16, planes)

def bn(planes):
    return nn.BatchNorm2d(planes)

def syncbn(planes):
    return nn.SyncBatchNorm(planes)

'''
    Author: Sungguk Cha
    FPN for semantic segmentation.
    Design is from "Panoptic Feature Pyramid Networks"
    utilizes output stride 32 backbones including efficientnet
'''

def conv3_gn_relu(inplanes, outplanes):
    res = nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(32, outplanes),
        nn.ReLU()
    )
    return res

def conv_gn_relu(inplanes, outplanes):
    res = nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0),
        nn.GroupNorm(32, outplanes),
        nn.ReLU()
    )
    return res

class _FPN(nn.Module):
    def __init__(self, args, num_classes):
        super(_FPN, self).__init__()

        self.resolution = (720, 1280)
        
        if args.norm == 'gn': norm=gn
        elif args.norm == 'bn': norm=bn
        elif args.norm == 'syncbn': norm=syncbn
        else:
            print(args.norm, "normalization is not implemented")
            raise NotImplementedError
        
        Fs = {
                'efficientnet-b7': (2560, 224, 80, 48), 
                'efficientnet-b6': (2304, 200, 72, 40),
                'efficientnet-b5': (2048, 176, 64, 40),
                'efficientnet-b4': (1792, 160, 56, 32)
            }

        F5, F4, F3, F2 = Fs[args.backbone]

        #
        self.f5 = conv_gn_relu(F5, 256)
        self.f4 = conv_gn_relu(F4, 256)
        self.f3 = conv_gn_relu(F3, 256)
        self.f2 = conv_gn_relu(F2, 256)
        #
        self.s51 = conv3_gn_relu(256, 256)
        self.s52 = conv3_gn_relu(256, 256)
        self.s53 = conv3_gn_relu(256, 128)
        self.s41 = conv3_gn_relu(256, 256)
        self.s42 = conv3_gn_relu(256, 128)
        self.s3 = conv3_gn_relu(256, 128)
        self.s2 = conv3_gn_relu(256, 128)
        #
        self.semantic_branch = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)


    def forward(self, features):
        f5 = features[3]
        f4 = features[2]
        f3 = features[1]
        f2 = features[0]

        p5 = self.f5(f5)
        p4 = self._upsample_add(p5, self.f4(f4))
        p3 = self._upsample_add(p4, self.f3(f3))
        p2 = self._upsample_add(p3, self.f2(f2))

        p5 = self.semantic_branch( self._upsample( self.s53( self._upsample( self.s52( self._upsample( self.s51( p5 ) ) ) ) ) ) )
        p4 = self.semantic_branch( self._upsample( self.s42( self._upsample( self.s41( p4 ) ) ) ) )
        p3 = self.semantic_branch( self._upsample( self.s3( p3 ) ) )
        p2 = self.semantic_branch( self.s2( p2 ) )

        x = self._upsample(p5 + p4 + p3 + p2, ratio=4)
        return x
        

    def _upsample(self, x, ratio=2):
        _, _, H, W = x.size()
        H *= ratio
        W *= ratio
        if H == 44: H = 45
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y


class FPN(nn.Module):
    def __init__(self, args, num_classes):
        super(FPN, self).__init__()
        self.backbone = build_backbone(args)
        self.fpn = _FPN(args, num_classes)

    def forward(self, input):
        features = self.backbone(input)
        x = self.fpn(features)
        return x

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.GroupNorm) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.fpn]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.GroupNorm) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

