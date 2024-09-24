"""
Utility functions
"""
import numpy as np
import cv2
from tqdm import tqdm
from numpy.lib import stride_tricks
import torch
import math
import argparse
import os


def get_parser(train=True):
    parser = argparse.ArgumentParser(description='Args for segmentation model training + testing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', type=str, default='test',
                        help='test mode')

    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100, help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=16, help='Batch size', dest='batch_size')
    parser.add_argument('--num_workers', type=int, default=1, help='Dataloader num_workers', dest='num_workers')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--min_lr', type=float, default=1e-5,help='Minimum learning rate', dest='min_lr')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay', dest='weight_decay')
    parser.add_argument('--factor', type=float, default=0.1, help='Factor', dest='factor')
    parser.add_argument('--patience', type=int, default=5, help='Patience', dest='patience')

    parser.add_argument('--height', type=int, default=512, help='Dimension Height', dest='height')
    parser.add_argument('--width', type=int, default=512, help='Dimension Width', dest='width')

    parser.add_argument('--labels', type=str, default='data/',
                        help='path to labels. will recursively search.')
    parser.add_argument('--inputs', '-in', type=str, default='data/', help='path to inputs')

    parser.add_argument('-f', '--load', dest='load', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.1,
                        help='Percent of the data that is used as validation (0-1.0)')
    parser.add_argument('-q', '--frequency', dest='freq', type=int, default=10,
                        help='save frequency of intermediate images') # no longer needed
    parser.add_argument('-n', '--name', dest='name', type=str, default='debugging', help='Name of the model')
    parser.add_argument('--save_dir', '-d', type=str, default='checkpoints', help='Base dir to save checkpoints and images')
    parser.add_argument('--smooth', action='store_true', help='apply morphological smoothing to prediction mask?')
    parser.add_argument('--net', '-ne', type=str, default='unet++',
                        help='Specify model type [unet | unet++ | densenet | unet3d]')
    parser.add_argument('--matte', '-m', action='store_true', help='append DenseBlock neural matte')
    parser.add_argument('--growth_rate', '-g', type=int, default=4, help='DenseBlock neural matte growth rate')
    parser.add_argument('--n_layers', '-nl', type=int, default=4, help='Number of layers in DenseBlock neural matte')
    parser.add_argument('--attn', action='store_true', help='add attention modules')

    parser.add_argument('--path', '-p', type=str, default=os.getcwd(),
                        help='Path to a text file containing lists of values for each hyperparameter to check')

    return parser


def psnr(img1, img2):
    """
	psnr = 20*log10((L-1)/rmse)
	"""

    mse = np.mean((img1 - img2) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / math.sqrt(mse))
    return psnr


def recon_and_rectify(patches, sz, w):
    ### RECONSTRUCTION
    recon = torch.zeros(sz)
    count = torch.zeros(sz)
    for i, p in enumerate(patches):
        recon[:, :, :, i:i + w] += p
        count[:, :, :, i:i + w] += 1

    recon /= count
    return recon


# https://stackoverflow.com/questions/42336919/how-to-extract-paches-from-3d-image-in-python
def create_sliding_window_tensor(data, blck, strd):
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    dims = np.r_[nbl, blck]
    data6 = stride_tricks.as_strided(data, strides=strides, shape=dims)
    return data6.reshape(-1, *blck)


def write_video(fname, frames, fps, shape):
    api = cv2.CAP_FFMPEG
    code = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    output = cv2.VideoWriter(fname, api, code, fps, shape)
    with tqdm(total=len(frames), desc='Writing') as pbar:
        for frame in frames:
            output.write(frame)
            pbar.update(1)
    output.release()
    # cv2.destroyAllWindows()

def manual_crop(frame, **kwargs):
    args = {}
    for key, value in kwargs.items():
        args[key] = value

    y = args['y']
    x = args['x']
    h = args['h']
    w = args['w']
    cropped = frame[y:y + h, x:x + w]
    return cropped


# traverses a sorted array for max path from start in a given direction
def __traverse(arr, start_i, start_v, max_gap, op):
    assert op == '+' or op == '-'

    i = start_i
    prev = start_v
    while i >= 0 and i < len(arr):
        curr = arr[i]

        gap = abs(curr - prev)
        if gap > max_gap:
            return prev
        else:
            prev = curr

        if op == '+':
            i += 1
        elif op == '-':
            i -= 1

    return prev


def __span(arr, center, max_gap):
    start_i = arr.index(center)
    return __traverse(arr, start_i, center, max_gap, '-'), __traverse(arr, start_i, center, max_gap, '+')


def center_color_crop(frame, **kwargs):
    args = {}
    for key, value in kwargs.items():
        args[key] = value

    threshold = args['threshold']
    max_gap = args['max_gap']

    # find all colored coordinates
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > threshold))

    # find the max spread from the center y-coord
    h, w, _ = gray.shape
    center = (int(h // 2), int(w // 2))
    y = list(coords[:, 0])
    y.sort()
    min_y, max_y = __span(y, center[0], max_gap)

    # re-threshold based on y crop and find the x span
    gray = gray[min_y:max_y + 1, :]
    h, w, _ = gray.shape
    center = (int(h // 2), int(w // 2))
    coords = np.column_stack(np.where(gray > threshold))
    x = list(coords[:, 1])
    x.sort()
    min_x, max_x = __span(x, center[1], max_gap)

    cropped = frame[min_y:max_y + 1, min_x:max_x + 1]
    return cropped


def contour_crop(frame, **kwargs):
    """
	From last answer to this Stack Overflow question: 
	https://stackoverflow.com/questions/28759253/how-to-crop-the-internal-area-of-a-contour	
	"""
    args = {}
    for key, value in kwargs.items():
        args[key] = value

    # find edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

    # find contours and sort by area
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # find bounding box and crop
    c = cnts[0]
    x, y, w, h = cv2.boundingRect(c)
    cropped = frame[y:y + h, x:x + w]

    return cropped
