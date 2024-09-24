from comet_ml import Experiment
import logging
import os
import sys
import cv2
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml
from models import get_net

from dataloader import *
from dataloader.StoneData import StoneData

from torch.utils.data import DataLoader

from util import   get_parser

from util.Segmentation import Segmentation
from albumentations.core.composition import Compose
from albumentations import Resize

from train import validate
import torch.backends.cudnn as cudnn


def get_args():
    parser = get_parser(train=False)
    return parser.parse_args()

def test_net(net,
              net_type,
              dataset,
              device,
              batch_size=1,
              img_scale=1,
              n_channels = 3,
              n_classes = 1,
              model_name='unet',
              save_dir='exps',
              smooth=False):


    config = yaml.safe_load(open('config.yaml'))
    experiment = Experiment(
        api_key=config['api_key'],
        project_name=config['project_name'],
        workspace=config['workspace']
    )

    # exp_name = os.path.basename(model_name)
    # exp_name, _ = os.path.splitext(exp_name)

    experiment.set_name(model_name)
    experiment.add_tag(net_type)
    with experiment.test():

        hyper_params = {
            'batch_size': batch_size,
            'img_scale': img_scale,
            'n_channels': n_channels,
            'n_classes': n_classes
        }

        experiment.log_parameters(hyper_params)

        n_test = len(dataset)
        print('NUM TEST: ', n_test)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

        activation = nn.Sigmoid()
        criterion = nn.BCEWithLogitsLoss(reduction='mean').cuda()

        logging.info(f'''Starting training:
            Batch size:      {batch_size}
            Device:          {device.type}
            Images scaling:  {img_scale}
        ''')

        net.eval()
        
        # path needs to exist for img to save
        # otherwise, OpenCV just returns False and doesn't raise an error
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # reuse validation fnc from training
        scores, to_save = validate(net,test_loader,device, activation, criterion)

        experiment.log_metric('Loss', sum(scores['loss']) / scores['n_batch'])
        # print(f'batch count {scores["n_batch"]}')
        experiment.log_metric('Loss std', np.std(scores['loss']))

        ### DICE
        val_score = sum(scores['dice']) / scores['n_batch']
        experiment.log_metric('Dice', val_score)
        experiment.log_metric('Dice std', np.std(scores['dice']))
        file_path = os.path.join(save_dir, 'dice_scores.txt')
        # print(val_score)
        # print(scores['dice'])
        with open(file_path, 'a') as file:
            for name, item in zip(scores['names'], scores['dice']):
                # for item in batch:
                # print('dice score: ', item)
                file.write(name + ' ' + str(item) + '\n')
        # print(f'Dice count {len(scores["dice"])}')
        # print(f'batch count {scores["n_batch"]}')

        ### PSNR
        psnr_score = sum(scores['psnr']) / scores['n_batch']
        experiment.log_metric('PSNR', psnr_score)

        ### ROC
        fpr_lst = [f for f, _, _ in scores['roc']]
        tpr_lst = [t for _, t, _ in scores['roc']]
        # roc_auc = sum([a for _, _, a in scores['roc']]) / scores['n_val']
        roc_auc = sum([a for _, _, a in scores['roc']]) / scores['n_batch']
        # print('roc')
        # print(len(scores['roc']))
        # print('n_val')
        # print(scores['n_val'])

        # https://stats.stackexchange.com/questions/186337/average-roc-for-repeated-10-fold-cross-validation-with-probability-estimates
        tprs = []
        base_fpr = np.linspace(0, 1, 101)


        fig, ax = plt.subplots(1, 1)

        for f, t in zip(fpr_lst, tpr_lst):
            ax.plot(f, t, 'b', alpha=0.15)
            tpr = np.interp(base_fpr, f, t)
            tpr[0] = 0.0
            tprs.append(tpr)

        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std

        # if math.isnan(roc_auc):
        #     print(f'scores[\'roc\']: ')
        #     print(scores['roc'])
        #     print(f'scores[\'n_val\']: ')
        #     print(scores['n_val'])

        ax.plot(base_fpr, mean_tprs, 'b', label='mean ROC curve (area = %0.2f)' % roc_auc)
        ax.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

        # print(scores['roc'])
        # temp for single batch
        # ax.plot(scores['roc'][0][0], scores['roc'][0][1], label='ROC curve (area = %0.2f)' % scores['roc'][0][2])

        ax.plot([0, 1], [0, 1], 'r--')
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        ax.set_aspect('equal', 'datalim')
        ax.set_title('ROC')
        ax.legend(loc="lower right")
        fig.savefig(os.path.join(save_dir, 'test_roc.jpg' ))
        experiment.log_metric('auc_roc', roc_auc)
        experiment.log_metric('auc_roc std', np.std([a for _, _, a in scores['roc']]))
        experiment.log_figure(figure=fig, figure_name='ROC for test of %s' % model_name)

        ### PIXEL ACC
        avg_tp = sum([tp for tp, _, _, _ in scores['acc']]) / scores['n_batch']
        avg_tn = sum([tn for _, tn, _, _ in scores['acc']]) / scores['n_batch']
        avg_fp = sum([fp for _, _, fp, _ in scores['acc']]) / scores['n_batch']
        avg_fn = sum([fn for _, _, _, fn in scores['acc']]) / scores['n_batch']
        total_tp = sum([tp for tp, _, _, _ in scores['acc']])
        total_tn = sum([tn for _, tn, _, _ in scores['acc']])
        total_fp = sum([fp for _, _, fp, _ in scores['acc']])
        total_fn = sum([fn for _, _, _, fn in scores['acc']])

        tpr = total_tp / (total_tp + total_fn)
        fpr = total_fp / (total_fp + total_tn)
        tnr = total_tn / (total_tn + total_fp)
        fnr = total_fn / (total_fn + total_tp)
        experiment.log_metric('TPR', tpr)
        experiment.log_metric('FPR', fpr)
        experiment.log_metric('TNR', tnr)
        experiment.log_metric('FNR', fnr)

        experiment.log_metric('Val Count', scores['n_val'])
        experiment.log_metric('Batch Count', scores['n_batch'])

        experiment.log_metric('TP', avg_tp)
        experiment.log_metric('TN', avg_tn)
        experiment.log_metric('FP', avg_fp)
        experiment.log_metric('FN', avg_fn)
        pix_acc = (avg_tp + avg_tn) / (avg_tp + avg_tn + avg_fp + avg_fn)
        experiment.log_metric('Pixel Acc', pix_acc)


        ### IoU
        avg_iou = sum(scores['iou']) / scores['n_batch']
        experiment.log_metric('IoU', avg_iou)
        experiment.log_metric('IoU std', np.std(scores['iou']))

        # tag high-performing models
        if val_score > 0.9:
            experiment.add_tag('>0.9')
        if val_score > 0.8:
            experiment.add_tag('>0.8')

        for o_ind, elem in enumerate(to_save):

            unmod_lst = [np.moveaxis(i, 0, -1) for i in copy.deepcopy(elem['unmod'])]

            pred_seg = Segmentation(copy.deepcopy(elem['unmod']), elem['output'], elem['names'])
            heatmap_seg = copy.deepcopy(pred_seg)

            masked_imgs = pred_seg.contour_mask(smooth=smooth)

            gt_seg = Segmentation(copy.deepcopy(elem['unmod']), elem['true'], elem['names'])
            gt_masked = gt_seg.apply_masks(smooth=smooth)

            for i_ind, (img, pred_map, unmod, gt, heat_map, name) in enumerate(zip(masked_imgs, pred_seg.masks, unmod_lst, gt_masked, heatmap_seg.masks,
                                                      elem['names'])):
                basename = os.path.basename(name)
                base, ext = os.path.splitext(basename)

                # hmaps
                scaled_map = (heat_map * 255).astype(np.uint8)
                hmap = cv2.applyColorMap(scaled_map, cv2.COLORMAP_JET)

                # combine and log
                combined = np.concatenate(
                    (unmod.astype(np.uint8), gt.astype(np.uint8), img.astype(np.uint8), hmap.astype(np.uint8)),
                    axis=1)
                im_path = os.path.join(save_dir,
                                       net_type + '_' + base + '_combined_%d_%d.jpg' % (o_ind,i_ind))

                cv2.imwrite(im_path, combined)
                # experiment.log_image(im_path, name=base + '_%d_%d' % (o_ind,i_ind), image_format='jpg')

    experiment.end()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # faster convolutions, but more memory
    cudnn.benchmark = True

    dataset = StoneData(args.inputs, args.labels, mode=args.mode,
                        transform=Compose([Resize(args.height, args.width)
                                                # ,transforms.Normalize()
                                        ])
                        # transform=v2.Compose([
                        #     v2.Resize(args.height, args.width),
                        #     v2.ToTensor(),
                        #     v2.Normalize([0.485, 0.456, 0.406],
                        #                  [0.229, 0.224, 0.225])
                        # ])
                        , args=args)

    net = get_net(args, device)

    try:
        test_net(net=net,
                  net_type=args.net.lower(),
                  dataset=dataset,
                  batch_size=args.batch_size,
                  device=device,
                  img_scale=args.scale,
                  model_name=args.name,
                  save_dir=args.save_dir+'/'+args.name,
                  smooth=args.smooth)
    except KeyboardInterrupt:
        logging.info('Testing interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
