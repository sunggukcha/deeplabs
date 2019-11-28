from modeling.backbone import resnet, xception, drn, mobilenet, wider_resnet, ibnnet
from modeling.backbone.efficientnet_pytorch.model import EfficientNet as efficientnet
import torch.nn as nn

def gn(planes):
	return nn.GroupNorm(16, planes)
def bn(planes):
        return nn.BatchNorm2d(planes)
def syncbn(planes):
        return nn.SyncBatchNorm(planes)

def build_backbone(args):
    Norm = args.norm
    backbone = args.backbone
    output_stride = args.output_stride

    if Norm == 'gn': norm=gn
    elif Norm == 'bn': norm=bn
    elif Norm == 'syncbn': norm=syncbn
    else:
        print (Norm, " <= normalization is not implemented")

    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, norm)
    elif backbone == 'resnet152':
        return resnet.ResNet152(output_stride, norm)
    elif backbone == 'wider_resnet':
        return wider_resnet.WiderResNet38(output_stride, norm, dec, abn)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, norm)
    elif backbone == 'drn':
        return drn.drn_d_38(norm)
    elif backbone == 'ibn':
        return ibnnet.resnet101_ibn_a(output_stride, norm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, norm)
    elif backbone.split('-')[0] == 'efficientnet':
        return efficientnet.from_pretrained(model_name=backbone, Norm=Norm, FPN=args.model=='fpn')
    else:
        raise NotImplementedError
