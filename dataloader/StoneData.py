import os
from os.path import join
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import json
import cv2
import copy
import warnings

from dataloader import *

warnings.filterwarnings("ignore")
plt.ion()


class StoneData(Dataset):

    # def __init__(self, file, root, mode='/train', transform=None, test=False):
    def __init__(self, pinputs='./data/', plabels='./data', mode='train', transform=None, json_lib=False, args=None):

        # set paths for images and labels
        # if mode == 'test' or mode == 'train' or mode == 'val':
        if mode is not None:
            label_images = glob.glob(join(pinputs, f'{mode}/images/**/*.PNG'), recursive=True)
            label_masks = glob.glob(join(plabels, f'{mode}/masks/**/*.PNG'), recursive=True)
        else:
            raise ValueError('Unknown arg %s' % mode)

        self.images = label_images
        self.masks = label_masks

        self.args = args
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # return data pair
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.images[idx]
        image = cv2.imread(self.images[idx])
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_UNCHANGED)

        # mask = np.zeros((image.shape[0], image.shape[1]))
        # mask = image.copy()
        # for region in self.annotated_regions[idx].keys():
        #     x = self.annotated_regions[idx][region]['shape_attributes']['all_points_x']
        #     y = self.annotated_regions[idx][region]['shape_attributes']['all_points_y']
        #
        #     points = np.array([x, y]).astype('int32').T
        #     cv2.fillConvexPoly(mask, points, 1.)


        #### RESIZE

        # image = cv2.resize(image, (448,448))
        # NORMALIZATION
        # normalization yields worse performance
        # image = (image - self.mean) / self.stddev

        # mask = cv2.resize(mask, (448,448))

        mask = np.expand_dims(mask, axis=2).astype(float)


        # if 'data/inputs/test/ww10212021_manual_crop/ww10212021_manual_crop_95.jpg' == img_name:
        #     print('nolabels')

        unmod = copy.deepcopy(image)

        # using albumentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if self.args:
            unmod = cv2.resize(unmod, (self.args.height,self.args.width))

        image = np.moveaxis(image, -1, 0)
        unmod = np.moveaxis(unmod, -1, 0)
        mask = np.moveaxis(mask, -1, 0)


        sample = {'unmod': torch.from_numpy(unmod).type(torch.FloatTensor),
                  'image': torch.from_numpy(image).type(torch.FloatTensor),
                  'mask': torch.from_numpy(mask).type(torch.FloatTensor),
                  'name': img_name}

        return sample

if __name__ == '__main__':
    dataset = StoneData(transform=transforms.Compose([ToTensor(), RandomColorJitter()]))
    dataset_get = dataset[1]
    # plt.imshow(dataset_get['mask'])
    # plt.savefig('test2.jpg')

    image = dataset_get['image'].numpy().transpose((1,2,0))
    unmod = dataset_get['unmod'].numpy().transpose((1,2,0))
    cv2.imwrite('summaries/figs/unmod.jpg', unmod)
    cv2.imwrite('summaries/figs/color_jitter.jpg', image)
