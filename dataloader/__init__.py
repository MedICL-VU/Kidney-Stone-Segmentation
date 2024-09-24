import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        unmod, image, mask, name = sample['unmod'], sample['image'], sample['mask'], sample['name']
        new_h, new_w = self.output_size, self.output_size
        img = image.reshape(new_h, new_w)
        return {'unmod': unmod,'image': img, 'mask': mask, 'name': name}


# class Normalize(object):
#     def __init__(self, inplace=False):
#         self.mean = (0.5692824, 0.55365936, 0.5400631)
#         self.std = (0.1325967, 0.1339596, 0.14305606)
#         self.inplace = inplace
#
#     def __call__(self, sample):
#         unmod, image, mask, name = sample['unmod'], sample['image'], sample['mask'], sample['name']
#         return {'image': TF.normalize(image, self.mean, self.std, self.inplace), 'mask': mask,
#                 'unmod': unmod, 'name':name}


class ToTensor(object):

    def __call__(self, sample):
        dtype = torch.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        unmod, image, mask, name = sample['unmod'], sample['image'], sample['mask'], sample['name']

        return {'unmod':torch.from_numpy(unmod).type(dtype),
                'image': torch.from_numpy(image).type(dtype),
                'mask': torch.from_numpy(mask).type(dtype),
                'name':name}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        unmod, image, mask, name = sample['unmod'], sample['image'], sample['mask'], sample['name']
        if random.random() < self.p:
            image = TF.hflip(image)

            mask = TF.hflip(mask)

        return {'unmod': unmod, 'image': image, 'mask': mask, 'name':name}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        unmod, image, mask, name = sample['unmod'], sample['image'], sample['mask'], sample['name']
        if random.random() < self.p:

            image = TF.vflip(image)

            mask = TF.vflip(mask)

        return {'unmod': unmod, 'image': image, 'mask': mask, 'name':name}

class RandomGaussianBlur(object):
    def __init__(self, p=0.2, kernel_size=[5,9], sigma=[0.1, 5]):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, sample):
        unmod, image, mask, name = sample['unmod'], sample['image'], sample['mask'], sample['name']
        if random.random() < self.p:

            image = TF.gaussian_blur(image, self.kernel_size, self.sigma)

        return {'unmod': unmod, 'image': image, 'mask': mask, 'name':name}

class RandomElasticTransform(object):
    def __init__(self, p=0.2, alpha=250.0):
        self.p = p
        self.elastictransform = transforms.ElasticTransform(alpha=alpha)

    def __call__(self, sample):
        unmod, image, mask, name = sample['unmod'], sample['image'], sample['mask'], sample['name']
        if random.random() < self.p:
            image = self.elastictransform(image)
        return {'unmod': unmod, 'image': image, 'mask': mask, 'name':name}

class RandomPosterize(object):
    def __init__(self, p=0.2, bits=2):
        self.p = p
        self.bits = bits

    def __call__(self, sample):
        unmod, image, mask, name = sample['unmod'], sample['image'], sample['mask'], sample['name']
        if random.random() < self.p:
            image = TF.posterize(image.type(torch.uint8), self.bits)
        return {'unmod': unmod, 'image': image.type(torch.float32), 'mask': mask, 'name':name}

class RandomSharpness(object):
    def __init__(self, p=0.2, sharpness_factor=(0.5, 1.5)):
        self.p = p
        self.sharpness_factor = sharpness_factor

    def __call__(self, sample):
        unmod, image, mask, name = sample['unmod'], sample['image'], sample['mask'], sample['name']
        if random.random() < self.p:
            sharpness = random.uniform(self.sharpness_factor[0], self.sharpness_factor[1])
            image = TF.adjust_sharpness(image, sharpness)
        return {'unmod': unmod, 'image': image, 'mask': mask, 'name': name}

class RandomEqualize(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, sample):
        unmod, image, mask, name = sample['unmod'], sample['image'], sample['mask'], sample['name']
        if random.random() < self.p:
            image = TF.equalize(image.type(torch.uint8))
        return {'unmod': unmod, 'image': image.type(torch.float32), 'mask': mask, 'name':name}

class RandomColorJitter(object):
    def __init__(self, p=0.2, brightness=(0.25, 1.25), contrast=(0.75, 1.25), saturation=(0.75, 1.25), hue=(-0.2, 0.2)):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        unmod, image, mask, name = sample['unmod'], sample['image'], sample['mask'], sample['name']
        if random.random() < self.p:
            # image *= 255.
            # image = Image.fromarray(np.uint8(image))
            modifications = []

            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            modifications.append(transforms.Lambda(lambda img: TF.adjust_brightness(image, brightness_factor)))

            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            modifications.append(transforms.Lambda(lambda img: TF.adjust_contrast(image, contrast_factor)))

            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            modifications.append(transforms.Lambda(lambda img: TF.adjust_saturation(image, saturation_factor)))

            hue_factor = random.uniform(self.hue[0], self.hue[1])
            modifications.append(transforms.Lambda(lambda img: TF.adjust_hue(image, hue_factor)))

            random.shuffle(modifications)
            modification = transforms.Compose(modifications)
            image = modification(image)

            # image = np.array(image)
            # image = np.float32(image) / 255.
        return {'unmod': unmod, 'image': image, 'mask': mask, 'name': name}


def calc_acc(input, output):
    sum = 0.0
    for i in range(input.shape[0]):
        sum += torch.sqrt((input[i][0] - output[i][0]) ** 2 + (input[i][1] - output[i][1]) ** 2)
    return sum / input.shape[0]
