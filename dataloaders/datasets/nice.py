import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class Nice(data.Dataset):
    '''
	BDD100k Drivable Area segmentation
    '''
    NUM_CLASSES = 3

    def __init__(self, args, root=Path.db_root_dir('nice')):

        self.root = root
        self.args = args
        self.files = []

        for img in os.listdir(root):
            if img.split('.')[-1] == 'jpg':
                self.files.append( os.path.join( root, os.path.basename(img) ) )

        self.void_classes = [] #[0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [0, 1, 2] #[7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['Not drivable', 'Drivable area', 'Alternative drivable area'] 

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        print("Found %d images" % (len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        img_path = self.files[index].rstrip()

        _img = Image.open(img_path).convert('RGB')
        _name = os.path.basename(img_path)

        sample = {'image': _img, 'name': _name}
        return self.transform(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform(self, sample):

        if self.args.nice:
            composed_transforms = transforms.Compose([
            	tr.Normalize(mean=(0.279, 0.293, 0.290), std=(0.197, 0.198, 0.201)),
	            #tr.Normalize(mean=(0.4620, 0.5392, 0.6004), std=(0.4603, 0.5377, 0.5989)),
	            tr.ToTensor()])
        else:
            composed_transforms = transforms.Compose([
                tr.Normalize(mean=(0.279, 0.293, 0.290), std=(0.197, 0.198, 0.201)),
                #tr.Normalize(mean=(0.4620, 0.5392, 0.6004), std=(0.4603, 0.5377, 0.5989)),
                tr.ToTensor()])

        return composed_transforms(sample)

