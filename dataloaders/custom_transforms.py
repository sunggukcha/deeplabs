import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        if 'label' in sample:
            return {'image': img, 'label': sample['label'], 'name': sample['name']}

        return {'image': img,
                'name' : sample['name']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()

        if 'label' in sample:
            mask = sample['label']
            mask = np.asarray(mask).astype(np.float32)
            mask = torch.from_numpy(mask).float()
            return {'image': img, 'label': mask, 'name': sample['name']}

        return {'image': img,
                'name' : sample['name']}



class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        if 'label' in sample:
            mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if 'label' in sample:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if 'label' in sample:
            return {'image': img, 'label': mask, 'name': sample['name']}

        return {'image': img,
                'name' : sample['name']}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        if 'label' in sample:
            mask = sample['label']
            mask = mask.rotate(rotate_degree, Image.NEAREST)
            return {'image': img, 'label': mask, 'name': sample['name']}

        return {'image': img,
                'name' : sample['name']}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        if 'label' in sample:
            return {'image': img, 'label': sample['label'], 'name': sample['name']}

        return {'image': img,
                'name' : sample['name']}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        if 'label' in sample:
            mask = sample['label']
            mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            if 'label' in sample:
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        if 'label' in sample:
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            return {'image': img, 'label': mask, 'name': sample['name']}

        return {'image': img,
                'name' : sample['name']}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        if 'label' in sample:
            mask = sample['label']
            mask = mask.resize((ow, oh), Image.NEAREST)
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            return {'image': img, 'label': mask, 'name': sample['name']}

        return {'image': img,
                'name' : sample['name']}

class RandomCrop2(object):
	'''
		Keeping BDD100k aspect ratio in training crops
		It takes great advantage in securing width-wise wide crop
	'''
	def __init__(self, crop_size):
		self.width	= crop_size
		self.height	= int(crop_size / 1.7777777777777777)
	def __call__(self, sample):
		img 	= sample['image']
		w, h	= img.size
		x	= random.randint(0, w - self.width)
		y	= random.randint(0, h - self.height)
		img	= img.crop( (x, y, x + self.width, y + self.height) )
		if 'label' in sample:
			label	= sample['label']
			label	= label.crop((x, y, x + self.width, y + self.height))
			return {'image': img, 'label': label, 'name': sample['name']}
		return {'image': img,
                	'name' : sample['name']}

'''
	Author: Sungguk Cha
	RandomCrop transform assumes 720p as input and 720 cropsize
	it randomly choose x value from [0, 1280-720) and return the cropped image
	it may reduce much much more time than FixScaleCrop above for the given assumption
'''

class RandomCrop(object):
	def __init__(self, crop_size):
		self.crop_size = crop_size
		assert self.crop_size == 720

	def __call__(self, sample):
		img = sample['image']
		w, h = img.size
		x = random.randint(0, w - self.crop_size)
		img = img.crop( (x, 0, x + self.crop_size, 720) )
		if 'label' in sample:
			label   = sample['label']
			label = label.crop( (x, 0, x + self.crop_size, 720) )
			return {'image': img, 'label': label, 'name': sample['name']}
		return {'image': img,
                        'name' : sample['name']}

		
		

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        if 'label' in sample:
            mask = sample['label']
            mask = mask.resize(self.size, Image.NEAREST)
            return {'image': img, 'label': mask, 'name': sample['name']}
        return {'image': img, 'name': sample['name']}
