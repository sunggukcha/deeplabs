import argparse
import os
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.visualize import Visualize as Vs
from torchsummary import summary

def gn(planes):
	return nn.GroupNorm(16, planes)
def syncbn(planes):
	pass
def bn(planes):
	return nn.BatchNorm2d(planes)
def syncabn(devices):
	return False
	def _syncabn(planes):
		return InplaceABNSync(planes, devices)
	return _syncabn
def abn(planes):
	return InPlaceABN(planes)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.vs = Vs()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        if self.args.norm == 'gn':
            norm = gn
        elif self.args.norm == 'bn':
            if self.args.sync_bn:
                norm = syncbn
            else:
                norm = bn
        elif self.args.norm == 'abn':
            if self.args.sync_bn:
                norm = syncabn(self.args.gpu_ids)
            else:
                norm = abn
        else:
            print("Please check the norm.")
            exit()

        # Define network
        if self.args.model	=='deeplabv3+':
             model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        Norm=args.norm,
                        freeze_bn=args.freeze_bn)
        elif self.args.model	=='deeplabv3':
             model = DeepLabv3(Norm=args.norm,
			backbone=args.backbone,
			output_stride=args.out_stride,
			num_classes=self.nclass,
			freeze_bn=args.freeze_bn)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model = model
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def test(self):
        self.model.eval()
        self.args.examine = False
        tbar = tqdm(self.test_loader, desc='\r')
        if self.args.color:
            __image = True
        else:
            __image = False
        for i, sample in enumerate(tbar):
            images = sample['image']
            names = sample['name']
            if self.args.cuda:
                images = images.cuda()
            with torch.no_grad():
                output = self.model(images)
            preds = output.data.cpu().numpy()
            preds = np.argmax(preds, axis=1)
            if __image:
                images  = images.cpu().numpy()
            if not self.args.color:
                self.vs.predict_id(preds, names, self.args.save_dir)
            else:
                self.vs.predict_color(preds, images, names, self.args.save_dir)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        if self.args.color or self.args.examine:
            __image = True
        else:
            __image = False
        for i, sample in enumerate(tbar):
            images, targets = sample['image'], sample['label']
            names = sample['name']
            if self.args.cuda:
                images, targets = images.cuda(), targets.cuda()
            with torch.no_grad():
                output = self.model(images)
            loss = self.criterion(output, targets)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            preds = output.data.cpu().numpy()
            targets = targets.cpu().numpy()
            preds = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(targets, preds)
            if __image:
                images  = images.cpu().numpy()
            if self.args.id:
                self.vs.predict_id(preds, names, self.args.save_dir)
            if self.args.color:
                self.vs.predict_color(preds, images, names, self.args.save_dir)
            if self.args.examine:
                self.vs.predict_examine(preds, targets, images, names, self.args.save_dir)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes', 'bdd'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=False,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # model
    parser.add_argument('--model', type=str, default='deeplabv3+',
			choices=['deeplabv3+', 'deeplabv3'])
    # Normalizations
    parser.add_argument('--norm', type=str, default='gn',
			choices=['gn', 'bn', 'abn'],
			help='normalization methods')

    # training hyper params
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--test', action='store_true', default=False,
			help='if true, inference test set')
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--save-dir', type=str, default='./prd',
			help='visualized image save directory')
    parser.add_argument('--id', action='store_true', default=False,
			help='save id images')
    parser.add_argument('--color', action='store_true', default=False,
			help='save visualized images')
    parser.add_argument('--examine', action='store_true', default=False,
			help='save visualized examine images')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = 1

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    if args.test:
        trainer.test()
    else:
        trainer.validation(0)

if __name__ == "__main__":
   main()
