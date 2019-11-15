from concurrent import futures
import nevergrad.optimization as optimization
from nevergrad import instrumentation as instru
import argparse
import torch
from train_loop import TrainLoop
import torch.optim as optim
import torch.utils.data
import model as model_
import numpy as np
from data_load import Loader
import os
import sys
from optimizer import TransformerOptimizer
from utils import *

def get_file_name(dir_):

	idx = np.random.randint(1)

	fname = dir_ + '/' + str(np.random.randint(1,999999999,1)[0]) + '.pt'

	while os.path.isfile(fname):
		fname = dir_ + '/' + str(np.random.randint(1,999999999,1)[0]) + '.pt'

	file_ = open(fname, 'wb')
	pickle.dump(None, file_)
	file_.close()

	return fname

# Training settings
parser=argparse.ArgumentParser(description='HP random search for fv antispoofing')
parser.add_argument('--input-size', type=int, default=2000, metavar='N', help='Input dimensionality (default: 2000)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--n-cycles', type=int, default=1, metavar='N', help='Number of cycles over train data to complete one epoch')
parser.add_argument('--train-hdf-path', type=str, default='./data/train.hdf', metavar='Path', help='Path to hdf data')
parser.add_argument('--valid-hdf-path', type=str, default='./data/test.hdf', metavar='Path', help='Path to hdf data')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--budget', type=int, default=30, metavar='N', help='Maximum training runs')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--hp-workers', type=int, help='number of search workers', default=1)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status. Default is 1')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--logdir', type=str, default=None, metavar='Path', help='Path for checkpointing')
args=parser.parse_args()
args.cuda=True if not args.no_cuda and torch.cuda.is_available() else False

def train(lr, l2, momentum, max_gnorm, warmup, input_size, n_hidden, hidden_size, dropout_prob, smoothing, n_cycles, epochs, batch_size, valid_batch_size, n_workers, cuda, train_hdf_path, valid_hdf_path, cp_path, logdir):

	hp_dict = {'lr':lr, 'l2':l2, 'momentum':momentum, 'max_gnorm':max_gnorm, 'warmup':warmup, 'input_size':input_size, 'n_hidden':n_hidden, 'hidden_size':hidden_size, 'dropout_prob':dropout_prob, 'smoothing':smoothing, 'n_cycles':n_cycles, 'epochs':epochs, 'batch_size':batch_size, 'valid_batch_size':valid_batch_size, 'n_workers':n_workers, 'cuda':cuda, 'train_hdf_path':train_hdf_path, 'valid_hdf_path':valid_hdf_path, 'cp_path':cp_path}

	cp_name = get_file_name(cp_path)

	if args.logdir:
		from torch.utils.tensorboard import SummaryWriter
		writer = SummaryWriter(log_dir=logdir+cp_name, purge_step=True)
		writer.add_hparams(hparam_dict=hp_dict, metric_dict={'best_eer':0.5})
	else:
		writer = None

	train_dataset = Loader(hdf5_clean = train_hdf_path+'train_clean.hdf', hdf5_attack = train_hdf_path+'train_attack.hdf', label_smoothing=smoothing, n_cycles=n_cycles)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

	valid_dataset = Loader(hdf5_clean = valid_hdf_path+'valid_clean.hdf', hdf5_attack = valid_hdf_path+'valid_attack.hdf', n_cycles=1)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=n_workers)

	model = model_.MLP(n_in=input_size, nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob)

	if cuda:
		device=get_freer_gpu()
		model=model.cuda(device)

	optimizer=TransformerOptimizer(optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2, nesterov=True), lr=lr, warmup_steps=warmup)

	trainer=TrainLoop(model, optimizer, train_loader, valid_loader, max_gnorm=max_gnorm, verbose=-1, cp_name=cp_name, save_cp=True, checkpoint_path=cp_path, cuda=cuda, logger=writer)

	for i in range(5):

		if i>0:
			print(' ')
			print('Trial {}'.format(i+1))
			print(' ')

		try:
			cost = trainer.train(n_epochs=epochs, save_every=epochs+10)

			print(' ')
			print('Best EER in file ' + cp_name + ' was: {}'.format(cost))
			print(' ')
			print('With hyperparameters:')
			print('Hidden layer size size: {}'.format(int(hidden_size)))
			print('Number of hidden layers: {}'.format(int(n_hidden)))
			print('Dropout rate: {}'.format(dropout_prob))
			print('Batch size: {}'.format(batch_size))
			print('LR: {}'.format(lr))
			print('Warmup iterations: {}'.format(warmup))
			print('Momentum: {}'.format(momentum))
			print('l2: {}'.format(l2))
			print('Max. Grad. norm: {}'.format(max_gnorm))
			print('Label smoothing: {}'.format(smoothing))
			print(' ')

			if args.logdir:
				writer.add_hparams(hparam_dict=hp_dict, metric_dict={'best_eer':cost})

			return cost
		except:
			pass

	print('Returning dummy cost due to failures while training.')
	cost=0.99
	if args.logdir:
		writer.add_hparams(hparam_dict=hp_dict, metric_dict={'best_eer':cost})
	return cost

lr=instru.var.OrderedDiscrete([1.0, 0.5, 0.1, 0.01])
l2=instru.var.OrderedDiscrete([0.01, 0.001, 0.0001, 0.00001])
momentum=instru.var.OrderedDiscrete([0.7, 0.85, 0.95])
max_gnorm=instru.var.OrderedDiscrete([10.0, 100.0, 1.0])
warmup=instru.var.OrderedDiscrete([1, 500, 2000, 5000])
input_size=args.input_size
n_hidden=instru.var.OrderedDiscrete([1, 2, 3, 4, 5, 6, 8, 10])
hidden_size=instru.var.OrderedDiscrete([64, 128, 256, 512, 1024])
dropout_prob=instru.var.OrderedDiscrete([0.01, 0.1, 0.2, 0.4])
smoothing=instru.var.OrderedDiscrete([0.0, 0.01, 0.1, 0.2])
n_cycles=args.n_cycles
epochs=args.epochs
batch_size=args.batch_size
valid_batch_size=args.valid_batch_size
n_workers=args.workers
cuda=args.cuda
train_hdf_path=args.train_hdf_path
valid_hdf_path=args.valid_hdf_path
cp_path=args.checkpoint_path
logdir=args.logdir

instrum=instru.Instrumentation(lr, l2, momentum, max_gnorm, warmup, input_size, n_hidden, hidden_size, dropout_prob, smoothing, n_cycles, epochs, batch_size, valid_batch_size, n_workers, cuda, train_hdf_path, valid_hdf_path, cp_path, logdir)

hp_optimizer=optimization.optimizerlib.RandomSearch(instrumentation=instrum, budget=args.budget, num_workers=args.hp_workers)

with futures.ThreadPoolExecutor(max_workers=args.hp_workers) as executor:
	print(hp_optimizer.optimize(train, executor=executor, verbosity=2))
