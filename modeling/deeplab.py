import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

def Norm(planes):
	return nn.GroupNorm(16, planes)

class DeepLabv3(nn.Module):
	def __init__(self, Norm, backbone='resnet', output_stride=16, num_classes=3, freeze_bn=False, abn=False):
		super(DeepLabv3, self).__init__()
		self.abn	= abn

		if backbone == 'drn':
			output_stride = 8

		self.backbone	= build_backbone(backbone, output_stride, Norm, dec=False, abn=abn)
		self.aspp	= build_aspp(backbone, output_stride, Norm, dec=False)
		if freeze_bn:
			self.freeze_bn()
	def forward(self, input):
		x	= self.backbone(input)
		x	= self.aspp(x)
		x	= F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
		return x
	def freeze_bn(self):
		'''
			not implemented freezing for abn(s) and gn
		'''
		for m in self.modules():
			if isinstance(m, SyncrhonizedBatchNorm2d):
				m.eval()
			elif isinstance(m, nn.BatchNorm2d):
				m.eval()

	def get_1x_lr_params(self):
		modules = [self.backbone]
		for i in range(len(modules)):
			for m in modules[i].named_modules():
				if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
						or isinstance(m[1], nn.BatchNorm2d):
					for p in m[1].parameters():
						if p.requires_grad:
							yield p

	def get_10x_lr_params(self):
		modules = [self.aspp]
		for i in range(len(modules)):
			for m in modules[i].named_modules():
				if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
						or isinstance(m[1], nn.BatchNorm2d):
					for p in m[1].parameters():
						if p.requires_grad:
							yield p


class DeepLab(nn.Module):
    def __init__(self, Norm, backbone='resnet', output_stride=16, num_classes=21,
                 freeze_bn=False, abn=False, deep_dec=True):
        super(DeepLab, self).__init__()
        self.abn	= abn
        self.deep_dec	= deep_dec # if True, it deeplabv3+, otherwise, deeplabv3

        if backbone == 'drn':
            output_stride = 8
        if backbone.split('-')[0] == 'efficientnet':
            output_stride = 32

        self.backbone = build_backbone(backbone, output_stride, Norm)
        self.aspp = build_aspp(backbone, output_stride, Norm)
        if self.deep_dec:
            self.decoder = build_decoder(num_classes, backbone, Norm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        if self.deep_dec:
            x = self.decoder(x, low_level_feat)
            x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        '''
           Sungguk comment
           as I am not freezing GN in training, it is not needed yet
           If I want to freeze, then I can list them like
              _list = [SyncrhonizedBatchNorm2d, nn.BatchNorm2d, nn.GroupNorm2d]
              for _i in _list:
                  if isinstance(m, _i):
                      m.eval()
           or just add an elif phrase
        '''
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


