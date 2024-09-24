from comet_ml import Experiment
import logging
import os
import sys
import cv2
import copy

import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import get_net
import yaml

from torch.utils.tensorboard import SummaryWriter
from dataloader import *
from dataloader.StoneData import StoneData


from torch.utils.data import DataLoader, random_split

from sklearn.metrics import roc_curve, auc, jaccard_score
from util import psnr, get_parser
from util.Dice import dice_coeff

from util.Segmentation import Segmentation
from util.History import History

from albumentations.core.composition import Compose
import albumentations as A


def get_args():
    parser = get_parser(train=True)
    return parser.parse_args()


def check_paths(save_dir):
    # path needs to exist for img to save
    # otherwise, OpenCV just returns False and doesn't raise an error
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, 'results')):
        os.makedirs(os.path.join(save_dir, 'results'))
    if not os.path.exists(os.path.join(save_dir, 'results/train_imgs')):
        os.makedirs(os.path.join(save_dir, 'results/train_imgs'))
    if not os.path.exists(os.path.join(save_dir, 'cp')):
        os.mkdir(os.path.join(save_dir, 'cp'))


def log_scores(args, scores, writer, optimizer, dice, experiment, global_step, epoch):
    experiment.log_metric('Loss', sum(scores['loss']) / scores['n_batch'], step=global_step, epoch=epoch)
    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

    ### DICE
    val_score = sum(scores['dice']) / scores['n_batch']
    experiment.log_metric('Dice', val_score, step=global_step, epoch=epoch)

    ### PSNR
    psnr_score = sum(scores['psnr']) / scores['n_batch']
    experiment.log_metric('PSNR', psnr_score, step=global_step, epoch=epoch)

    ### ROC
    fpr_lst = [f for f, _, _ in scores['roc']]
    tpr_lst = [t for _, t, _ in scores['roc']]
    roc_auc = sum([a for _, _, a in scores['roc']]) / scores['n_batch']

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

    ax.plot(base_fpr, mean_tprs, 'b', label='mean ROC curve (area = %0.2f)' % roc_auc)
    ax.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.set_aspect('equal', 'datalim')
    ax.set_title('ROC at epoch %d' % (epoch + 1))
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(args.save_dir + '/' + args.name, 'results/train_imgs', 'roc_%d.jpg' % (epoch + 1)))
    experiment.log_figure(figure=fig, figure_name='ROC at epoch %d' % (epoch + 1))

    ### PIXEL ACC
    avg_tp = sum([tp for tp, _, _, _ in scores['acc']]) / scores['n_batch']
    avg_tn = sum([tn for _, tn, _, _ in scores['acc']]) / scores['n_batch']
    avg_fp = sum([fp for _, _, fp, _ in scores['acc']]) / scores['n_batch']
    avg_fn = sum([fn for _, _, _, fn in scores['acc']]) / scores['n_batch']

    experiment.log_metric('TP', avg_tp, step=global_step, epoch=epoch)
    experiment.log_metric('TN', avg_tn, step=global_step, epoch=epoch)
    experiment.log_metric('FP', avg_fp, step=global_step, epoch=epoch)
    experiment.log_metric('FN', avg_fn, step=global_step, epoch=epoch)
    pix_acc = (avg_tp + avg_tn) / (avg_tp + avg_tn + avg_fp + avg_fn)
    experiment.log_metric('Pixel Acc', pix_acc, step=global_step, epoch=epoch)

    ### IoU
    avg_iou = sum(scores['iou']) / scores['n_batch']
    experiment.log_metric('IoU', avg_iou, step=global_step, epoch=epoch)

    # tag high-performing models
    if val_score > 0.9:
        experiment.add_tag('>0.9')
    if val_score > 0.8:
        experiment.add_tag('>0.8')

    logging.info('Validation Dice Coeff: {}'.format(val_score))
    writer.add_scalar('Dice/test', val_score, global_step)

    # log score to History object
    dice.log(val_score)

    plt.close('all')  # ran into some bugs without, try checking it later


def save(args, to_save, save_dir, epoch):
    for elem in to_save:

        unmod_lst = [np.moveaxis(i, 0, -1) for i in copy.deepcopy(elem['unmod'])]

        pred_seg = Segmentation(copy.deepcopy(elem['unmod']), elem['output'], elem['names'])
        heatmap_seg = copy.deepcopy(pred_seg)
        masked_imgs = pred_seg.contour_mask(smooth=args.smooth)

        gt_seg = Segmentation(copy.deepcopy(elem['input']), elem['true'], elem['names'])
        gt_masked = gt_seg.apply_masks(smooth=args.smooth)

        for img, pred_map, unmod, gt, heat_map, name in zip(masked_imgs, pred_seg.masks, unmod_lst, gt_masked,
                                                            heatmap_seg.masks, elem['names']):
            basename = os.path.basename(name)
            base, ext = os.path.splitext(basename)

            # hmaps
            scaled_map = (heat_map * 255).astype(np.uint8)
            hmap = cv2.applyColorMap(scaled_map, cv2.COLORMAP_JET)

            # combine and log
            combined = np.concatenate((
                unmod.astype(np.uint8), gt.astype(np.uint8), img.astype(np.uint8),
                hmap.astype(np.uint8)), axis=1)
            im_path = os.path.join(save_dir, 'results/train_imgs/',
                                   args.net.lower() + '_' + base + '_combined_%d.jpg' % (epoch))
            cv2.imwrite(im_path, combined)

    torch.save(net.state_dict(),
               os.path.join(os.path.join(save_dir, 'cp'), f'{args.name}_CP_epoch{epoch + 1}.pth'))
    logging.info(f'Checkpoint {epoch + 1} saved !')


def train(args, train_loader, model, criterion, optimizer, writer, experiment, global_step, epoch):
    model.train()
    pbar = tqdm(total=len(train_loader))
    for batch in train_loader:
        input = batch['image'].to(device=device, dtype=torch.float32)
        target = batch['mask'].to(device=device,
                                  dtype=torch.float32)  # mask_type = torch.float32 if n_classes == 1 else torch.long

        output = model(input)
        loss = criterion(output, target)  # BCE with logits implicitly applies sigmoid

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), global_step)
        experiment.log_metric('Loss', loss.item(), step=global_step, epoch=epoch)
        pbar.set_postfix(**{'loss (batch)': loss.item()})
        pbar.update(1)
        global_step += 1
    pbar.close()
    return global_step


def validate(net, loader, device, activation, criterion):
    """Evaluation without the densecrf with the dice coefficient"""

    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    to_save = []

    scores = {
        'n_val': 0,
        'n_batch': len(loader),
        'dice': [],
        'roc': [],
        'psnr': [],
        'acc': [],
        'iou': [],
        'loss': [],
        'names': []
    }
    with torch.no_grad():
        with tqdm(total=n_val + 1, desc='Validation round', unit='batch', leave=False) as pbar:
            for i, (batch) in enumerate(loader):

                unmod, input, target, names = batch['unmod'], batch['image'], batch['mask'], batch['name']
                input = input.to(device=device, dtype=torch.float32)
                target = target.to(device=device, dtype=mask_type)

                output = net(input)
                loss = criterion(output, target)
                output = activation(
                    output)  # BCE with logits implicitly applies sigmoid so left out from underlying model

                input = input.detach().cpu()
                target = target.detach().cpu()
                output = output.detach().cpu()

                to_save += [{'unmod': unmod.numpy(), 'input': input.numpy(),
                             'output': output.numpy(), 'true': target.numpy(), 'names': names}]

                scores['n_val'] += input.size(0)
                scores['loss'] += [loss.item()]

                pred_labels = (output > 0.5).float()

                scores['dice'] += [dice_coeff(pred_labels, target).item()]

                ### AS NUMPY MATRICES FROM HERE
                output = output.numpy()
                target = target.numpy()

                ### PSNR
                scores['psnr'] += [psnr(output, target)]

                ### ROC-AUC
                # https://www.kaggle.com/kmader/use-roc-curves-to-evaluate-segmentation-methods
                fpr, tpr, _ = roc_curve(target.astype(int).ravel(), output.ravel())
                roc_auc = auc(fpr, tpr)
                if np.isnan(roc_auc):
                    pass
                # print('nan val, throwing out this roc (typically seen when no positive elements in gt == meaningless)')
                else:
                    scores['roc'] += [(fpr, tpr, roc_auc)]

                ### PIXEL ACC
                pred_labels = pred_labels.byte().numpy()
                tp = np.sum(np.logical_and(pred_labels == 1, target == 1))
                tn = np.sum(np.logical_and(pred_labels == 0, target == 0))
                fp = np.sum(np.logical_and(pred_labels == 1, target == 0))
                fn = np.sum(np.logical_and(pred_labels == 0, target == 1))

                scores['acc'] += [(tp, tn, fp, fn)]

                ### IoU
                scores['iou'] += [jaccard_score(target.astype(int).ravel(), pred_labels.ravel())]

                scores['names']+= names

                pbar.update(1)
            pbar.close()
        return scores, to_save


def train_net(net,
              dataset,
              valset,
              device,
              epochs=20,
              batch_size=1,
              lr=0.0001,
              val_percent=0.1,
              save_cp=True,
              save_frequency=10,
              n_channels=3,
              n_classes=1,
              model_name='unet',
              save_dir='exps',
              smooth=False):
    #####################################################
    ###     https://www.comet.com/lu-d/tumorseg       ###
    #####################################################

    config = yaml.safe_load(open('config.yaml'))
    experiment = Experiment(
        api_key=config['api_key'],
        project_name=config['project_name'],
        workspace=config['workspace']
    )
    experiment.set_name(model_name)
    experiment.add_tag(args.net.lower())
    with experiment.train():

        hyper_params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'validation_percent': val_percent,
            'n_channels': n_channels,
            'n_classes': n_classes
        }

        experiment.log_parameters(hyper_params)

        n_val = len(valset)
        n_train = len(dataset)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

        writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
        global_step = 0

        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_cp}
            Device:          {device.type}
        ''')

        # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        params = filter(lambda p: p.requires_grad, net.parameters())  # added from
        optimizer = optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        activation = nn.Sigmoid()
        criterion = nn.BCEWithLogitsLoss(reduction='mean').cuda()

        dice = History(score='Dice', model_name=model_name)

        check_paths(save_dir)

        best_dice = 0
        for epoch in range(epochs):
            experiment.log_metric('LR', scheduler.get_last_lr(), step=global_step, epoch=epoch)
            global_step = train(args, train_loader, net, criterion, optimizer, writer, experiment, global_step, epoch)

            with experiment.validate():
                scores, to_save = validate(net, val_loader, device, activation, criterion)
                log_scores(args, scores, writer, optimizer, dice, experiment, global_step, epoch)

            scheduler.step()
            # scheduler.step(sum(scores['loss'])/scores['n_batch']) # for reducelronplateau
            # scheduler.step(sum(scores['dice'])/scores['n_batch'])
            if save_cp and sum(scores['dice']) / scores['n_batch'] > best_dice:
                save(args, to_save, save_dir, epoch)
                best_dice = sum(scores['dice']) / scores['n_batch']
            ### summarization
            dice.write()

            torch.cuda.empty_cache()
        # cv2.destroyAllWindows() # some versions need it to run headless

        experiment.end()
        writer.close()
        return dice


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    print('Building dataset...')


    dataset = StoneData(args.inputs, args.labels,
                        transform=Compose([
                            A.Resize(args.height, args.width),
                            # InsertRandomStone(stone_dir='./data/stone_for_augmentation', max_stones=50, base_size=30,
                            #                   p=0.5),
                            A.RandomRotate90(),
                            A.Flip(p=0.5),
                            # A.ColorJitter(p=0.25),
                            A.GaussianBlur(p=0.25),
                            # A.Posterize(p=0.25),
                            # A.Sharpen(p=0.25),
                            # A.Equalize(p=0.25),
                            # A.RandomBrightnessContrast(p=0.25),
                            # A.Perspective(p=0.25),
                            # A.ElasticTransform(p=0.25),
                            # transforms.Normalize()
                        ])

                        , args=args)

    valset = StoneData(args.inputs, args.labels, mode='val',
                       transform=Compose([
                           A.Resize(args.height, args.width)
                           ,
                           # transforms.Normalize()
                       ])
                       , args=args)

    net = get_net(args, device)

    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        print('Training...')
        train_net(net=net,
                  dataset=dataset,
                  valset=valset,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  device=device,
                  val_percent=args.val,
                  save_cp=True,
                  save_frequency=args.freq,
                  model_name=args.name,
                  save_dir=args.save_dir + '/' + args.name,
                  smooth=args.smooth)
    except KeyboardInterrupt:
        print('Interrupted...')
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        torch.cuda.empty_cache()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
