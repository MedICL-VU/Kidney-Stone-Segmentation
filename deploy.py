import os
import cv2
import numpy as np
import logging
import torch
from util.Segmentation import Segmentation
from util import contour_crop, write_video, get_parser
from model import get_net
import torch.nn as nn
from time import time

sigmoid = nn.Sigmoid()

def get_args():
	parser = get_parser(train=False)

	# diagnostic options for device detection and port specification
	parser.add_argument('--list', '-l', action='store_true',
						help='List ports')
	parser.add_argument('--port', '-o', type=int, default=0,
						help='Set video port, default=0 (usually builtin webcam on mac)')

	# options for processing stream through model
	parser.add_argument('--process', action='store_true',
						help='Process video stream from port through model')

	return parser.parse_args()

# https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
def list_ports():
	"""
	Test the ports and returns a tuple with the available ports and the ones that are working.
	"""
	non_working_ports = []
	dev_port = 0
	working_ports = []
	available_ports = []
	while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing.
		camera = cv2.VideoCapture(dev_port)
		if not camera.isOpened():
			non_working_ports.append(dev_port)
			print("Port %s is not working." %dev_port)
		else:
			is_reading, img = camera.read()
			w = camera.get(3)
			h = camera.get(4)
			if is_reading:
				print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
				working_ports.append(dev_port)
			else:
				print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
				available_ports.append(dev_port)
		dev_port +=1
	return available_ports,working_ports,non_working_ports

# https://answers.opencv.org/question/229567/reliable-video-capture-in-opencv-from-usb-video-capture-devices/
def stream(port):
	"""
	Test the video stream for port detection and selection.
	"""
	cap = cv2.VideoCapture(port)
	fps = cap.get(cv2.CAP_PROP_FPS)
	print('Reading video from port', port, 'with fps', fps)

	# cap.open(0, cv2.CAP_DSHOW)
	while cap.isOpened():
		ret, frame = cap.read()

		if ret:
			cv2.imshow('endo', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):  # imshow window must be selected --> press q to quit
			break

	cap.release()
	cv2.destroyAllWindows()

def deploy(args):
	"""
	Process frames from the real-time stream.
	"""
	port = args.port
	cap = cv2.VideoCapture(port)
	fps = cap.get(cv2.CAP_PROP_FPS)
	print('Reading video from port', port, 'with fps', fps)

	net_basename = os.path.basename(args.load)
	net_base, net_ext = os.path.splitext(net_basename)

	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

	# make dirs
	save_dir = os.path.join(args.save_dir, net_base)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	cap_dir = os.path.join(save_dir, 'cap')
	if not os.path.exists(cap_dir):
		os.makedirs(cap_dir)
	img_dir = os.path.join(save_dir, 'img')
	if not os.path.exists(img_dir):
		os.makedirs(img_dir)
	anno_dir = os.path.join(save_dir, 'anno')
	if not os.path.exists(anno_dir):
		os.makedirs(anno_dir)
	hmap_dir = os.path.join(save_dir, 'hmap')
	if not os.path.exists(hmap_dir):
		os.makedirs(hmap_dir)
	comb_dir = os.path.join(save_dir, 'combined')
	if not os.path.exists(comb_dir):
		os.makedirs(comb_dir)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# retrieve net
	net = get_net(args, device)

	# set to evaluation mode
	net.eval()

	mean = cv2.imread('summaries/mean.jpg')
	stddev = cv2.imread('summaries/stddev.jpg')

	combined_frames = []
	count = 0
	start = time()
	while cap.isOpened():
		ret, capt = cap.read()

		# save every capture
		cv2.imwrite(os.path.join(cap_dir, 'cap_' + str(count + 1) + '.jpg'), capt)

		if ret:

			try:
				image = contour_crop(capt)
				image = cv2.resize(image, (448, 448))
			except IndexError:
				continue

			unmod = np.copy(image)

			### MUST NORMALIZE IMAGES IF MODEL TRAINS ON NORMALIZED IMAGES
			# o/w synthesis won't work -> check in FrameDataset
			# normalization yields worse performance
			# image = (image - mean) / stddev

			dtype = torch.FloatTensor # if torch.cuda.is_available() else torch.FloatTensor
			unmod = unmod.transpose((2, 0, 1))
			image = image.transpose((2, 0, 1))
			unmod = np.expand_dims(unmod, axis=0)
			image = np.expand_dims(image, axis=0)

			image = torch.from_numpy(image).type(dtype)

			img = image.to(device=device, dtype=torch.float32)

			with torch.no_grad():
				# since BCE with logits loss implicitly applies sigmoid
				# it was left out of underlying trained model -> apply at end to obtain probabilities
				pred = net(img)
				pred = sigmoid(pred)

			pred = pred.detach().cpu().numpy()

			unmod_lst = [np.moveaxis(i, 0, -1) for i in np.copy(unmod)]

			pred_seg = Segmentation(np.copy(unmod), pred, None)
			masked_imgs = pred_seg.apply_masks(smooth=args.smooth)

			for masked_img, pred_map, unmod in zip(masked_imgs, pred_seg.masks, unmod_lst):

				# hmaps
				scaled_map = (pred_map * 255).astype(np.uint8)
				hmap = cv2.applyColorMap(scaled_map, cv2.COLORMAP_JET)

				# combine and log
				combined = np.concatenate(
					(unmod.astype(np.uint8), masked_img.astype(np.uint8), hmap.astype(np.uint8)),
					axis=1)
				combined_frames += [combined]
				cv2.imwrite(os.path.join(comb_dir, 'combined_' + str(count + 1) + '.jpg'), combined)
				cv2.imshow('combined', combined)  # can't show if running on ACCRE

				# saving every frame above
				cv2.imwrite(os.path.join(img_dir, 'img_' + str(count + 1) + '.jpg'),
							unmod)
				cv2.imwrite(os.path.join(anno_dir,
										 'anno_' + str(count+ 1) + '.jpg'), masked_img)
				cv2.imwrite(os.path.join(hmap_dir,
										 'hmap_'  + str(count + 1) + '.jpg'), hmap)

			count += 1

		if cv2.waitKey(1) & 0xFF == ord('q'):  # imshow window must be selected --> press q to quit
			print('Exiting video capture...')
			cap.release()
			cv2.destroyAllWindows()
			break

	runtime = time() - start
	net_fps = count / runtime
	print('Runtime = ', runtime)
	print('Process FPS = ', net_fps)

	# attempt to write video of combined frames
	write_video(os.path.join(save_dir, args.net + '_combined'), combined_frames, net_fps, (448 * 3, 448))


if __name__ == '__main__':
	args = get_args()

	if args.list:
		available, working, nonworking = list_ports()
		print()

	if args.process:
		print('Deploying...')
		deploy(args)
	else:
		print('Streaming...')
		stream(args.port)

