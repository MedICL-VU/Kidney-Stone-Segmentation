import os
from glob import glob
import cv2
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool

from util.Segmentation import Segmentation

def isSequential(foldername):

    return False

def genEmpty(image):
    # return same dimension array of false values
    return np.zeros(image.shape)

def binarize(mask):
    # reduces 3 channel into 1
    shape = mask.shape
    # shape[2] = 1
    binary = np.zeros((shape[0], shape[1]))
    binary[mask[:, :, 1] == 255] = 1.
    return binary

def listEmpty():
    folders = glob('data/expertstudy/*/')
    for folder in tqdm(folders):
        if not os.path.isdir(folder + 'SegmentationClass/'):
            print(f'No segmentations: {folder}')
            continue

def genEmptyMasks(skip=10):
    # checks masks for every 10 frames. If None, create an emptymask
    folders = glob('data/oracle_cvat/*/')

    if not os.path.isdir('data/images/'):
        os.mkdir('data/images/')
    if not os.path.isdir('data/masks/'):
        os.mkdir('data/masks/')

    for folder in tqdm(folders):
        if not os.path.isdir(folder + 'SegmentationClass/'):
            print(f'No segmentations: {folder}')
            continue

        foldername = os.path.basename(os.path.normpath(folder))
        taskname = foldername.split('.')[0][5:].strip()

        os.mkdir('data/images/'+taskname)
        os.mkdir('data/masks/' + taskname)

        # with open('tasks.txt', 'a') as f:
        #     f.write(f'{foldername},{isSequential(foldername)}\n')

        # print(f'foldername: {foldername}')

        images = sorted(glob(folder+ 'JPEGImages/*.PNG'))
        masks = sorted(glob(folder + 'SegmentationClass/*.png'))

        if isSequential(foldername):
            frameNames = [os.path.basename(f) for f in images]
        else:
            frameNames = [os.path.basename(f) for f in (images[::skip])]

        maskNames = [os.path.basename(f) for f in masks]

        for frame in frameNames:
            image = cv2.imread(folder + 'JPEGImages/' + frame)
            if frame.lower() in maskNames:
                mask = cv2.imread(folder + 'SegmentationClass/' + frame[:-3] + 'png')
            else:
                mask = genEmpty(image)

            mask = binarize(mask)

            cv2.imwrite(f'data/images/{taskname}/{frame}', image)
            cv2.imwrite(f'data/masks/{taskname}/{frame}', mask)

            # to read
            # cv2.imread('*.png', cv2.IMREAD_UNCHANGED)
        # break
    # print(len(folders))


def gen_expert_masks(skip=1):
    # checks masks for every 10 frames. If None, create an emptymask

    if not os.path.isdir('./data/images/'):
        os.mkdir('./data/images/')
    if not os.path.isdir('./data/masks/'):
        os.mkdir('./data/masks/')
    folders = glob('./data/_expertstudy/*/')

    for folder in tqdm(folders):
        if not os.path.isdir(folder + 'SegmentationClass/'):
            print(f'No segmentations: {folder}')
            continue

        foldername = os.path.basename(os.path.normpath(folder))
        # taskname = foldername.split('.')[0][5:].strip()
        taskname=foldername

        if not os.path.isdir(f'data/{taskname}'):
            os.mkdir(f'data/{taskname}')
        os.mkdir(f'data/{taskname}/images/')
        os.mkdir(f'data/{taskname}/masks/')

        # with open('tasks.txt', 'a') as f:
        #     f.write(f'{foldername},{isSequential(foldername)}\n')

        # print(f'foldername: {foldername}')

        images = sorted(glob(folder+ 'JPEGImages/*.PNG'))
        masks = sorted(glob(folder + 'SegmentationClass/*.png'))

        if isSequential(foldername):
            frameNames = [os.path.basename(f) for f in images]
        else:
            frameNames = [os.path.basename(f) for f in (images[::skip])]

        maskNames = [os.path.basename(f) for f in masks]

        for frame in frameNames:
            image = cv2.imread(folder + 'JPEGImages/' + frame)
            if frame.lower() in maskNames:
                mask = cv2.imread(folder + 'SegmentationClass/' + frame[:-3] + 'png')
            else:
                mask = genEmpty(image)

            mask = binarize(mask)

            cv2.imwrite(f'data/{taskname}/images/{frame}', image)
            cv2.imwrite(f'data/{taskname}/masks/{frame}', mask)

            # to read
            # cv2.imread('*.png', cv2.IMREAD_UNCHANGED)
        # break
    # print(len(folders))

def video_viz(folder):
    foldername = os.path.basename(os.path.normpath(folder))

    masks = sorted(glob(folder + '/*.PNG'))
    maskNames = [os.path.basename(f) for f in masks]

    if not os.path.isdir('data/viz/' + foldername):
        os.mkdir('data/viz/' + foldername)
    for frame in maskNames:
        image = cv2.imread(f'data/images/{foldername}/{frame}')
        mask = cv2.imread(f'data/masks/{foldername}/{frame}', cv2.IMREAD_UNCHANGED)

        newframe = Segmentation.static_contour(image, mask)

        cv2.imwrite(f'data/viz/{foldername}/{frame}', newframe)
    return True

def visualize_masks():
    if not os.path.isdir('data/viz/'):
        os.mkdir('data/viz/')
    folders = glob('data/expert*')

    # cpucount = os.cpu_count()
    # print(f'Available processes {cpucount}')
    # print(f'Using {cpucount -1}')

    # work = folders
    #
    # p = Pool(cpucount - 1)
    # results = p.imap(video_viz, work)
    for folder in tqdm(folders):
        foldername = os.path.basename(os.path.normpath(folder))

        masks = sorted(glob(folder + '/masks/*.PNG'))
        maskNames = [os.path.basename(f) for f in masks]

        if not os.path.isdir('data/viz/' + foldername):
            os.mkdir('data/viz/' + foldername)
        for frame in maskNames:
            image = cv2.imread(f'data/{foldername}/images/{frame}')
            mask = cv2.imread(f'data/{foldername}/masks/{frame}', cv2.IMREAD_UNCHANGED)

            newframe = Segmentation.static_contour(image, mask)

            cv2.imwrite(f'data/viz/{foldername}/{frame}', newframe)

def vid_crop(folder):
    foldername = os.path.basename(os.path.normpath(folder))

    masks = sorted(glob(folder + '/*.PNG'))
    maskNames = [os.path.basename(f) for f in masks]

    if not os.path.isdir('data/cropped/images' + foldername):
        os.mkdir('data/cropped/images/' + foldername)
    if not os.path.isdir('data/cropped/masks' + foldername):
        os.mkdir('data/cropped/masks/' + foldername)

    for frame in maskNames:
        image = cv2.imread(f'data/images/{foldername}/{frame}')
        mask = cv2.imread(f'data/masks/{foldername}/{frame}', cv2.IMREAD_UNCHANGED)

        newimage, newmask = contour_crop(image, mask)

        cv2.imwrite(f'data/cropped/images/{foldername}/{frame}', newimage)
        cv2.imwrite(f'data/cropped/masks/{foldername}/{frame}', newmask)


def crop_raw_data():
    if not os.path.isdir('data/cropped/'):
        os.mkdir('data/cropped/')
        os.mkdir('data/cropped/images/')
        os.mkdir('data/cropped/masks/')
    folders = glob('data/masks/*')
    # folders = ['data/masks/expert 1-2023_10_11_22_34_56-segmentation mask 1', 'data/masks/expert 2-2023_10_12_16_43_56-segmentation mask 1', 'data/masks/expert 3-2023_10_13_16_48_10-segmentation mask 1']
    # cpucount = os.cpu_count()
    # print(f'Available processes {cpucount}')
    # print(f'Using {cpucount -1}')
    #
    # work = folders
    #
    # p = Pool(cpucount - 1)
    # results = p.imap(vid_crop, work)

    for folder in tqdm(folders):
        foldername = os.path.basename(os.path.normpath(folder))

        masks = sorted(glob(folder + '/*.PNG'))
        maskNames = [os.path.basename(f) for f in masks]

        if not os.path.isdir('data/cropped/images' + foldername):
            os.mkdir('data/cropped/images/' + foldername)
        if not os.path.isdir('data/cropped/masks' + foldername):
            os.mkdir('data/cropped/masks/' + foldername)

        for frame in maskNames:
            image = cv2.imread(f'data/images/{foldername}/{frame}')
            mask = cv2.imread(f'data/masks/{foldername}/{frame}', cv2.IMREAD_UNCHANGED)

            newimage, newmask = contour_crop(image,mask)

            cv2.imwrite(f'data/cropped/images/{foldername}/{frame}', newimage)
            cv2.imwrite(f'data/cropped/masks/{foldername}/{frame}', newmask)

def contour_crop(frame, mask, **kwargs):
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
    # check bounds, select max
    boxsize = max(h,w)
    if boxsize + y >= len(frame[0]) or boxsize + x >= len(frame[1]):
        boxsize = min(len(frame[0]) - y, len(frame[1])-x)
    h = w = boxsize
    cropped_image = frame[y:y + h, x:x + w]
    cropped_mask = mask[y:y + h, x:x + w]

    return cropped_image, cropped_mask