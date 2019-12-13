'''
    author: Sungguk Cha
    email : navinad@naver.com

    evaluation script with label and prediction(image form)
'''

import argparse
import numpy as np
import os

from PIL import Image
from torch.utils import data
from tqdm import tqdm
from utils.metrics import Evaluator

class Loader(data.Dataset):
    def __init__(self, args):
        self.args = args

        # labels
        self.label_dir = args.labels
        '''
        for img in os.listdir(args.labels):
            _dir = os.path.join(args.labels, img)
            if os.path.isfile(_dir):
                self.labels.append(_dir)
        '''

        # predictions
        self.preds = []
        for img in os.listdir(args.preds):
            _dir = os.path.join(args.preds, img)
            if os.path.isfile(_dir):
                self.preds.append(_dir)

        #assert len(self.labels) == len(self.preds)

    def __len__(self):
        return len(self.preds)

    def __getitem__(self, index):
        _pred   = self.preds[index]
        _name   = os.path.basename(_pred.split('.')[-2] + '.png')
        _label  = os.path.join(self.label_dir, _name)
        assert os.path.basename(_pred.split('.')[-2]) == os.path.basename(_label.split('.')[-2])
        _label  = Image.open(_label).convert('RGB')
        _pred   = Image.open(_pred).convert('RGB')
        if _label.size != _pred.size:
            _pred = _pred.resize( _label.size, Image.BILINEAR )
        _label  = np.asarray(_label)
        _pred   = np.asarray(_pred)

        return {'pred':_pred, 'label':_label, 'name':_name}

class Eval(object):
    def __init__(self, args):
        self.args = args
        self.evaluator = Evaluator(args.nclass)
        self.loader = Loader(args)

    def evaluation(self):
        self.evaluator.reset()
        tbar = tqdm(self.loader)
        for i, sample in enumerate(tbar):
            names   = sample['name']
            preds   = sample['pred']
            labels  = sample['label']
            self.evaluator.add_batch(labels, preds)
        
        miou = self.evaluator.Mean_Intersection_over_Union()
        fwiou = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print("mIoU:", miou)
        print("fwIoU:", fwiou)

def get_args():
    parser = argparse.ArgumentParser()
    # Dataset specific
    parser.add_argument('--nclass', type=int)
    # Dataloader specific
    parser.add_argument('--preds', type=str)
    parser.add_argument('--labels', type=str)
    parser.add_argument('--vis', type=bool, default=False, action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    ev = Eval(args)
    ev.evaluation()
