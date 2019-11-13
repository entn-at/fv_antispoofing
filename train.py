from __future__ import print_function
import argparse
import torch
from train_loop import TrainLoop
import torch.optim as optim
import torch.utils.data
import model as model_
import numpy as np
from data_load import Loader
from torch.utils.tensorboard import SummaryWriter
from optimizer import TransformerOptimizer
from utils import *

# Training settings
parser = argparse.ArgumentParser(description='Antispoofing for utterance level representations of speech')
parser.add_argument('--input-size', type=int, default=2000, metavar='N', help='Input dimensionality (default: 2000)')
parser.add_argument('--hidden-size', type=int, default=512, metavar='S', help='latent layer dimension (default: 512)')
parser.add_argument('--n-hidden', type=int, default=1, metavar='N', help='maximum number of frames per utterance (default: 1)')
parser.add_argument('--dropout-prob', type=float, default=0.25, metavar='p', help='Dropout probability (default: 0.25)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=64, metavar='N', help='input batch size for validation (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='alpha', help='Alpha (default: 0.9)')
parser.add_argument('--l2', type=float, default=1e-5, metavar='L2', help='Weight decay coefficient (default: 0.00001)')
parser.add_argument('--max-gnorm', type=float, default=10., metavar='clip', help='Max gradient norm (default: 10.0)')
parser.add_argument('--warmup', type=int, default=4000, metavar='N', help='Iterations until reach lr (default: 4000)')
parser.add_argument('--smoothing', type=float, default=0.2, metavar='l', help='Label smoothing (default: 0.2)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--pretrained-path', type=str, default=None, metavar='Path', help='Path for pre trained model')
parser.add_argument('--logdir', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--train-hdf-path', type=str, default='./data/train.hdf', metavar='Path', help='Path to hdf data')
parser.add_argument('--valid-hdf-path', type=str, default=None, metavar='Path', help='Path to hdf data')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status. Default is 1')
parser.add_argument('--n-cycles', type=int, default=1, metavar='N', help='Number of cycles over train data to complete one epoch')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

if args.cuda:
	device = get_freer_gpu()
else:
	device = torch.device('cpu')

if args.logdir:
	writer = SummaryWriter(log_dir=args.logdir, purge_step=True)
else:
	writer = None

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

train_dataset = Loader(hdf5_clean = args.train_hdf_path+'train_clean.hdf', hdf5_attack = args.train_hdf_path+'train_attack.hdf', label_smoothing=args.smoothing, n_cycles=args.n_cycles)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

valid_dataset = Loader(hdf5_clean = args.valid_hdf_path+'valid_clean.hdf', hdf5_attack = args.valid_hdf_path+'valid_attack.hdf', n_cycles=1)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.workers)

model = model_.MLP(n_in=args.input_size, nh=args.n_hidden, n_h=args.hidden_size, dropout_prob=args.dropout_prob)

if args.pretrained_path is not None:
	ckpt = torch.load(args.pretrained_path, map_location = lambda storage, loc: storage)
	model = model_.MLP(n_in=ckpt['input_size'], nh=ckpt['n_hidden'], n_h=ckpt['hidden_size'], dropout_prob=ckpt['dropout_prob'])
	try:
		model.load_state_dict(ckpt['model_state'], strict=True)
	except RuntimeError as err:
		print("Runtime Error: {0}".format(err))
		model.load_state_dict(ckpt['model_state'], strict=False)
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

model = model.to(device)

optimizer = TransformerOptimizer(optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2, nesterov=True), lr=args.lr, warmup_steps=args.warmup)

trainer = TrainLoop(model, optimizer, train_loader, valid_loader, max_gnorm=args.max_gnorm, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda, logger=writer)

print('Cuda Mode: {}'.format(args.cuda))
print('Device: {}'.format(device))
print('Batch size: {}'.format(args.batch_size))
print('Validation batch size: {}'.format(args.valid_batch_size))
print('LR: {}'.format(args.lr))
print('Momentum: {}'.format(args.momentum))
print('l2: {}'.format(args.l2))
print('Max. grad norm: {}'.format(args.max_gnorm))
print('Warmup iterations: {}'.format(args.warmup))
print('Inputs dimensionality: {}'.format(args.input_size))
print('Number of hidden layers: {}'.format(args.n_hidden))
print('Size of hidden layers: {}'.format(args.hidden_size))
print('Label smoothing: {}'.format(args.smoothing))
trainer.train(n_epochs=args.epochs, save_every=args.save_every)
