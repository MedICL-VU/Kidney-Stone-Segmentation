"""
Segmentation [class]
Expects PyTorch batch shapes i.e. (batch_size,n_channels,*img.shape)
Directly overlays the predicted segmentations on the corresponding images

__init__:
imgs = tensor of input images (INPUT PARAM MODIFIED)
masks = tensor of predicted segmentations
names = ordered names of the inputs

apply_masks = overlays the masks directly on the images
"""
import os
import numpy as np
import cv2


class Segmentation:
    def __init__(self, imgs, masks, names):
        self.imgs = [img for img in imgs]
        self.masks = [np.squeeze(mask) for mask in masks]
        self.names = names

    def apply_masks(self, smooth=False):

        masked_imgs = []
        for img, mask in zip(self.imgs, self.masks):

            masked_img = None
            channels = [c for c in img]
            for i, channel in enumerate(channels):

                if i == 0:
                    channel[np.where(mask >= 0.5)] = 255
                    masked_img = channel
                else:
                    channel[np.where(mask >= 0.5)] = 0
                    masked_img = np.dstack([masked_img, channel]) if not masked_img is None else channel
                # if i == 0:
                #     channel[np.where(mask >= 0.5)] = channel[np.where(mask >= 0.5)] + 10
                #     masked_img = channel
                # else:
                #     channel[np.where(mask >= 0.5)] = 0
                #     masked_img = np.dstack([masked_img, channel]) if not masked_img is None else channel

            masked_imgs += [masked_img]

        # https://stackoverflow.com/questions/53877035/how-can-i-smooth-the-segmented-blob/64056653#64056653
        masked_imgs = [cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) for
                       mask in masked_imgs] if smooth else masked_imgs

        return masked_imgs

    def soft_masks(self, smooth=False):

        masked_imgs = []
        for img, mask in zip(self.imgs, self.masks):

            masked_img = None
            channels = [c for c in img]
            for i, channel in enumerate(channels):

                if i == 0:
                    channel[np.where(mask >= 0.5)] = channel[np.where(mask >= 0.5)] + 100
                    # channel[np.where(mask < 0.5)] = channel[np.where(mask < 0.5)] - 50
                    masked_img = channel
                else:
                    # channel[np.where(mask >= 0.5)] = 0

                    masked_img = np.dstack([masked_img, channel]) if not masked_img is None else channel

            masked_imgs += [masked_img]

        # https://stackoverflow.com/questions/53877035/how-can-i-smooth-the-segmented-blob/64056653#64056653
        masked_imgs = [cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) for
                       mask in masked_imgs] if smooth else masked_imgs

        return masked_imgs

    def contour_mask(self, smooth=False):

        masked_imgs = []
        for img, mask in zip(self.imgs, self.masks):

            masked_img = None
            channels = [c for c in img]

            # apply soft blue mask
            for i, channel in enumerate(channels):
                if i == 0:
                    # channel[np.where(mask >= 0.5)] = channel[np.where(mask >= 0.5)] + 100
                    # channel[np.where(mask < 0.5)] = channel[np.where(mask < 0.5)] - 50
                    masked_img = channel
                else:
                    # channel[np.where(mask >= 0.5)] = 0
                    masked_img = np.dstack([masked_img, channel]) if not masked_img is None else channel

            # mask to 3d for cv2
            # full_mask = np.zeros(list(masked_img.shape))
            # full_mask[:,:,0][np.where(mask >= 0.5)] = 255.
            # full_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask[np.where(mask>=0.5)] = 255
            mask[np.where(mask<1)] = 0
            pass
            # apply contour
            contours, heirarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(masked_img, contours, -1, (0,255,0), 3)

            masked_imgs += [masked_img]

        # https://stackoverflow.com/questions/53877035/how-can-i-smooth-the-segmented-blob/64056653#64056653
        masked_imgs = [cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) for
                       mask in masked_imgs] if smooth else masked_imgs

        return masked_imgs

    @staticmethod
    def static_contour(img, mask, smooth=False):

        masked_img = None
        channels = [img[:,:,0], img[:,:,1], img[:,:,2]]

        # apply soft blue mask
        for i, channel in enumerate(channels):
            if i == 0:
                # channel[np.where(mask >= 0.5)] = channel[np.where(mask >= 0.5)] + 100
                # channel[np.where(mask < 0.5)] = channel[np.where(mask < 0.5)] - 50
                masked_img = channel
            else:
                # channel[np.where(mask >= 0.5)] = 0
                masked_img = np.dstack([masked_img, channel]) if not masked_img is None else channel

        # mask to 3d for cv2
        # full_mask = np.zeros(list(masked_img.shape))
        # full_mask[:,:,0][np.where(mask >= 0.5)] = 255.
        # full_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask[np.where(mask>=0.5)] = 255
        mask[np.where(mask<1)] = 0
        pass
        # apply contour
        contours, heirarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(masked_img, contours, -1, (0,255,0), 3)

        # https://stackoverflow.com/questions/53877035/how-can-i-smooth-the-segmented-blob/64056653#64056653
        masked_img = cv2.morphologyEx(masked_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) if smooth else masked_img

        return masked_img

    def overlay_heatmaps(self):
        pass
