'''
	Author: Sungguk Cha
	eMail : navinad@naver.com
	It loads several models and stacks the prediction results.
'''

import argparse
import os
from tqdm import tqdm
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.visualize import Visualize as vs
from torchsummary import summary
import numpy as np
import torch.nn as nn

blows = 0

def gn(planes):
	return nn.GroupNorm(16, planes)

def blow(image, _class):
	'''
	post process subfunction
	blows '_class' class from an image
	'''
	global blows
	blows += 1
	image[image == _class] = 0
	return image

def post1(inputs):
	'''
	Post processing 1.
	It blows up classes less than {(0, 1): 1337, (0, 1, 2): [1597, 1304], (0, 2): 2836}
	'''
	results = []
	blowed = False
	for result in inputs:
		unique, counts = np.unique(result, return_counts=True)
		dic = dict(zip(unique, counts))
		unique = tuple(unique)
		if unique == (0, 1):
			if dic[1] < 1337:
				result = blow(result, 1)
				blowed = True
		elif unique == (0, 2):
			if dic[2] < 2836:
				result = blow(result, 2)
				blowed = True
		elif unique == (0, 1, 2):
			if dic[1] < 1597:
				result = blow(result, 1)
				blowed = True
			if dic[2] < 1304:
				result = blow(result, 2)
				blowed = True
		results.append(result)
	return results, blowed

class Stack(object):
	def __init__(self, args):
		self.args = args
		self.vs = vs(args.nice)

		#Dataloader
		kwargs = {"num_workers": args.workers, 'pin_memory': True}
		if self.args.dataset == 'bdd':
			_, _, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
		else: #self.args.dataset == 'nice':
			self.test_loader, self.nclass = make_data_loader(args, **kwargs)
		#else:
		#	raise NotImplementedError

		### Load models
		#backs = ["resnet", "resnet152"]
		backs = ["resnet", "ibn", "resnet152"]
		check = './ckpt'
		checks = ["herbrand.pth.tar", "ign85.12.pth.tar", "r152_85.20.pth.tar"]
		self.models = []
		self.M = len(backs)
		# define models
		for i in range(self.M):
			model = DeepLab(num_classes = self.nclass,
					backbone=backs[i],
					output_stride=16,
					Norm=gn,
					freeze_bn=False)
			self.models.append(model)
			self.models[i] = torch.nn.DataParallel(self.models[i], device_ids=self.args.gpu_ids)
			patch_replication_callback(self.models[i])
			self.models[i] = self.models[i].cuda()
		# load checkpoints
		for i in range(self.M):
			resume = os.path.join(check, checks[i])
			if not os.path.isfile( resume ):
				raise RuntimeError("=> no checkpoint found at '{}'".format(resume))
			checkpoint = torch.load( resume )
			dicts = checkpoint['state_dict']
			model_dict = {}
			state_dict = self.models[i].module.state_dict()
			for k, v in dicts.items():
				if k in state_dict:
					model_dict[k] = v
			state_dict.update(model_dict)
			self.models[i].module.load_state_dict(state_dict)
			print( "{} loaded successfully".format(checks[i]) )

	def predict(self, mode):
		for i in range(self.M):
			self.models[i].eval()
		tbar = tqdm(self.test_loader, desc='\r')
		for i, sample in enumerate(tbar):
			images = sample['image']
			names = sample['name']
			images.cuda()
			outputs = []
			with torch.no_grad():
				for i in range(self.M):
					output = self.models[i](images)
					output = output.data.cpu().numpy()
					outputs.append( output )
			if mode == "stack":
				results = outputs[0]
				for output in outputs[1:]:
					results += output
				results = np.argmax( results, axis=1 )
			if self.args.post:
				posts, blowed = post1(np.array(results))
				if blowed:
					images = images.cpu().numpy()
					self.vs.predict_color( results, images, names, self.args.savedir )
					_names = []
					for name in names:
						_name = name.split('.')[0] + "-blow.png"
						_names.append(_name)
					self.vs.predict_color( posts, images, _names, self.args.savedir )
				continue # saving blows
			if self.args.color:
				images = images.cpu().numpy()
				self.vs.predict_color( results, images, names, self.args.savedir )
			else:
				self.vs.predict_id( results, names, self.args.savedir )
		if self.args.post:
			global blows
			print(blows, "blows happened")
def get_args():
	parser = argparse.ArgumentParser()
	# Dataloader
	parser.add_argument('--dataset', default='bdd')
	parser.add_argument('--workers', type=int, default=0, metavar='N', help='dataloader threads')
	parser.add_argument("--img_list", default=None)
	parser.add_argument("--batch_size")
	# Model load
	parser.add_argument('--gpu_ids', type=str, default='0')
	# Prediction save
	parser.add_argument('--savedir', type=str, default='./prd')
	parser.add_argument('--color', default=False, action='store_true')
	parser.add_argument('--nice', default=False, action='store_true', help="Use nice RGB mean & std")
	parser.add_argument('--post', default=False, action='store_true', help="Activate post process")
	return parser.parse_args()

if __name__ == "__main__":
	args = get_args()
	args.batch_size = int(args.batch_size)
	args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
	stack = Stack(args)
	stack.predict(mode="stack")
