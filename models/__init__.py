import logging
import torch
import segmentation_models_pytorch as smp
from models.unext import UNext, UNext_S
from models.unet import UNet
import torch.nn as nn




def get_net(args, device):
	"""
	Creates the neural net given an argparse object, sends it to device, and loads if necessary.

	Parameters:
		args = argparse object

	Returns:
		net = specified neural net
	"""
	print('Building: ', args.net)

	if args.net.lower() == 'unet++':
		if args.attn:
			net = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3,
			                       classes=1,
			                       decoder_attention_type='scse')
		else:
			net = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3,
			                       classes=1)
	elif args.net.lower() == 'unet':
		if args.attn:
			net = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, decoder_attention_type='scse')
		else:
			net = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1)
	elif args.net.lower() == 'unext':
		net = UNext(num_classes=1, img_size=args.height)
	elif args.net.lower() == 'unexts':
		net = UNext_S(num_classes=1, img_size=args.height)
	elif args.net.lower() == 'unet_custom':
		net = UNet(n_channels=3, n_classes=1)
	else:
		raise ValueError('Unknown model type: %s', args.net)

	# attempt load if spec'd
	if args.load:
		net.load_state_dict(
			torch.load(args.load, map_location=device)
		)
		logging.info(f'Model loaded from {args.load}')

	print('Sending %s to device' % args.net, device)
	net.to(device=device)

	return net
