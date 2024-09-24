import logging
import torch
import segmentation_models_pytorch as smp
from model.densenet import FCDenseNet67, DenseBlock
from model.unet3d import UNet3D
from model.PraNet_Res2Net import PraNet
from model.unext import UNext, UNext_S
from model.unet import UNet
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
	elif args.net.lower() == 'densenet':
		net = FCDenseNet67(1)
	elif args.net.lower() == 'unet3d':
		net = UNet3D(in_channels=3, out_channels=1)
	elif args.net.lower() == 'unet':
		if args.attn:
			net = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, decoder_attention_type='scse')
		else:
			net = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1)
	elif args.net.lower() == 'pranet':
		net = PraNet(channel=32)
	elif args.net.lower() == 'unext':
		net = UNext(num_classes=1, img_size=args.height)
	elif args.net.lower() == 'unexts':
		net = UNext_S(num_classes=1, img_size=args.height)
	elif args.net.lower() == 'unet_custom':
		net = UNet(n_channels=3, n_classes=1)
	else:
		raise ValueError('Unknown model type: %s', args.net)

	if args.matte and not args.net.lower() == 'unet3d':
		growth_rate = args.growth_rate
		n_layers = args.n_layers

		print('Adding matte with growth %d and %d layers' % (growth_rate, n_layers))

		matte = DenseBlock(in_channels=1, growth_rate=growth_rate, n_layers=n_layers)

		# out layers from matte = 1 channel from base net + growth_rate * n_layers channels
		final_conv = nn.Conv2d(in_channels=1 + growth_rate * n_layers, out_channels=1, kernel_size=1, stride=1,
		                       padding=0, bias=True)
		net = nn.Sequential(net, matte, final_conv)

	# attempt load if spec'd
	if args.load:
		net.load_state_dict(
			torch.load(args.load, map_location=device)
		)
		logging.info(f'Model loaded from {args.load}')

	print('Sending %s to device' % args.net, device)
	net.to(device=device)

	return net
