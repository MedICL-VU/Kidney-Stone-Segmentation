from comet_ml import Experiment
from tqdm import tqdm
import logging
import os
import sys
import csv
import itertools
import cv2
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import get_net
# from model.PraNet.lib.PraNet_Res2Net import PraNet

from dataloader import *
from dataloader.StoneData import StoneData

from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
from util import get_parser, psnr

from util.Segmentation import Segmentation
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import Resize
# from torchvision.transforms import v2

import torch.backends.cudnn as cudnn
from util.Dice import dice_coeff

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'


def get_args():
    parser = get_parser(train=False)
    return parser.parse_args()


def predict(net, loader, device, activation):
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    predictions = []
    image_names = []

    with torch.no_grad():
        with tqdm(total=n_val + 1, desc='Generating predictions', unit='batch', leave=False) as pbar:
            for i, (batch) in enumerate(loader):
                unmod, input, target, names = batch['unmod'], batch['image'], batch['mask'], batch['name']
                input = input.to(device=device, dtype=torch.float32)
                target = target.to(device=device, dtype=mask_type)

                output = net(input)
                output = activation(
                    output)  # BCE with logits implicitly applies sigmoid so left out from underlying model

                output = output.detach().cpu()

                pred_labels = (output > 0.5).float()

                predictions.append(pred_labels)
                image_names.append(os.path.split(names[0])[1])

                pbar.update(1)
            pbar.close()
        return predictions, image_names


def save_to_csv(data, combo_names, image_names, filename):
    # Adding headers based on the length of the first inner list
    headers = [name for name in image_names]
    headers.insert(0, '')
    headers.insert(1, '')

    # Open the file and write the data
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)

        # Writing the header
        writer.writerow(headers)

        # Writing the data rows
        for row, combo in zip(data, combo_names):
            row.insert(0, combo[0])
            row.insert(1, combo[1])
            writer.writerow(row)

def save_to_csv2(data, combo_names, image_names, filename):
    # Adding headers based on the length of the first inner list
    headers = [name for name in image_names]
    headers.insert(0, '')

    # Open the file and write the data
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)

        # Writing the header
        writer.writerow(headers)

        # Writing the data rows
        for row, combo in zip(data, combo_names):
            row.insert(0, combo)
            writer.writerow(row)

def eval_net(net,
             net_type,
             datasets,
             sets,
             device,
             batch_size=1,
             img_scale=1,
             n_channels=3,
             n_classes=1,
             model_name='unet',
             save_dir='exps',
             smooth=False):
    # exp_name = os.path.basename(model_name)
    # exp_name, _ = os.path.splitext(exp_name)

    hyper_params = {
        'batch_size': batch_size,
        'img_scale': img_scale,
        'n_channels': n_channels,
        'n_classes': n_classes
    }
    set_idx = list(range(len(sets)+1))
    combinations = list(itertools.combinations(set_idx, 2))
    n_test = len(datasets[0])
    loaders = [DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True) for dataset in datasets]

    activation = nn.Sigmoid()
    net.eval()

    # get prediction masks
    predictions, image_names = predict(net, loaders[0], device, activation)

    # extract label_masks
    label_masks = [[],[],[],[],[],[],[], [], predictions]
    # label_masks = [[], predictions]


    # sets.append('model')
    # save_mask_dir = './data/model_predictions/masks'
    #
    # for image, name in zip(predictions, image_names):
    #     print(os.path.join(save_mask_dir, name))
    #     im = image[0].numpy()
    #     im = np.moveaxis(im, 0, -1)
    #     # print(im.shape)
    #     cv2.imwrite(os.path.join(save_mask_dir, name), im)


    for i, loader in enumerate(loaders):
        masks =[]

        for j, (batch) in enumerate(loader):
            unmod, input, target, names = batch['unmod'], batch['image'], batch['mask'], batch['name']
            target = target.to(dtype=torch.float32)
            masks.append(target)

        label_masks[i] = masks



    # linear_stone_pred = []
    # for idx, label in tqdm(enumerate(label_masks)):
    #     pred_vals = []
    #
    #     for mask in label:
    #         mask = mask.numpy()
    #         # dice_vals.append(dice_coeff(mask0, mask1).item())
    #         if np.sum(mask) != 0:
    #             pred_vals.append(1)
    #         else:
    #             pred_vals.append(0)
    #
    #     linear_stone_pred.append(pred_vals)


    # print('Saving Linear')
    # save outputs to csv
    # save_to_csv2(linear_stone_pred, [0,1,2,3,4,5,6], image_names, 'summaries/tables/expert_dice_stone_presence.csv')
    # save_to_csv2(linear_stone_pred, [0, 1, 2, 3, 4, 5, 6], image_names,
    #              'summaries/tables/expert_dice_stone_presence.csv')

    dice_scores = []
    psnr_scores = []
    iou_scores = []

    for combo in tqdm(combinations):
        dice_vals = []
        psnr_vals = []
        iou_vals = []

        idx0, idx1 = combo

        for mask0, mask1 in zip(label_masks[idx0], label_masks[idx1]):
            dice_vals.append(dice_coeff(mask0, mask1).item())
            mask0 = mask0.numpy()
            mask1 = mask1.numpy()
            psnr_vals.append(psnr(mask0, mask1))
            iou_vals.append(jaccard_score(mask0.astype(int).ravel(), mask1.ravel()))

        dice_scores.append(dice_vals)
        psnr_scores.append(psnr_vals)
        iou_scores.append( iou_vals)

    sets.append('model')
    combo_names = [(sets[i], sets[j]) for i, j in combinations]

    print('Saving Combos')
    # save outputs to csv
    save_to_csv(dice_scores, combo_names, image_names, 'summaries/tables/expert_dice_staple_oracle.csv')
    # save_to_csv(iou_scores, combo_names, image_names, 'summaries/tables/expert_iou3.csv')
    # save_to_csv(psnr_scores, combo_names, image_names, 'summaries/tables/expert_psnr3.csv')




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    # net = UNet(n_channels=3, n_classes=1, bilinear=True)
    #
    # net = FCDenseNet67(1)

    # net = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1)
    # net = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1)
    # net = smp.DeepLabV3(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1)
    # os.chdir('..') # FOR ACCRE IF RUNNING FROM SLURM DIR

    # faster convolutions, but more memory
    cudnn.benchmark = True
    expert_sets = ['expert_staple', 'expert0', 'expert1', 'expert2', 'expert3', 'expert4', 'expert5', 'oracle']

    datasets = [
        StoneData(args.inputs, args.labels, mode=expert,
                  transform=Compose([Resize(args.height, args.width)
                                     # ,transforms.Normalize()
                                     ]) )
        for expert in expert_sets
    ]


    net = get_net(args, device)

    try:
        eval_net(net=net,
                 net_type=args.net.lower(),
                 datasets=datasets,
                 sets=expert_sets,
                 batch_size=args.batch_size,
                 device=device,
                 img_scale=args.scale,
                 model_name=args.name,
                 save_dir=args.save_dir + '/' + args.name,
                 smooth=args.smooth)
    except KeyboardInterrupt:
        logging.info('Evaluation interrupt')
        try:
            torch.cuda.empty_cache()
            sys.exit(0)
        except SystemExit:
            os._exit(0)
