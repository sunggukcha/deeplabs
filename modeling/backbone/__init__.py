from modeling.backbone import resnet, xception, drn, mobilenet, wider_resnet, ibnnet
import torch.nn as nn

def Norm(planes):
	return nn.GroupNorm(32, planes)

def build_backbone(backbone, output_stride, BatchNorm, dec=True, abn=False):
    if BatchNorm == None:
        BatchNorm = Norm

    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'resnet152':
        return resnet.ResNet152(output_stride, BatchNorm)
    elif backbone == 'wider_resnet':
        return wider_resnet.WiderResNet38(output_stride, BatchNorm, dec, abn)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_38(BatchNorm)
    elif backbone == 'ibn':
        return ibnnet.resnet101_ibn_a(output_stride, BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
