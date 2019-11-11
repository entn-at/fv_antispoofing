import torch
import torch.nn.functional as F

import numpy as np

import os
from tqdm import tqdm

import model as model_
from utils import compute_eer

class TrainLoop(object):

	def __init__(self, model, optimizer, train_loader, valid_loader, max_gnorm, verbose=-1, cp_name=None, save_cp=False, checkpoint_path=None, checkpoint_epoch=None, cuda=True, logger=None):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, cp_name) if cp_name else os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.model = model
		self.max_gnorm = max_gnorm
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.verbose = verbose
		self.save_cp = save_cp
		self.total_iters = 0
		self.cur_epoch = 0
		self.device = next(self.model.parameters()).device
		self.logger = logger

		if self.valid_loader is not None:
			self.history = {'train_loss': [], 'train_loss_batch': [], 'valid_loss': []}
		else:
			self.history = {'train_loss': [], 'train_loss_batch': []}

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=1, save_every=1):

		while (self.cur_epoch < n_epochs):
			if self.verbose>0:
				print(' ')
				print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
				train_iter = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
			else:
				train_iter = enumerate(self.train_loader)

			train_loss_epoch=0.0

			for t, batch in train_iter:
				train_loss = self.train_step(batch)
				self.history['train_loss_batch'].append(train_loss)
				train_loss_epoch+=train_loss
				if self.logger:
					self.logger.add_scalar('Train Loss', train_loss, self.total_iters)
					self.logger.add_scalar('Info/LR', self.optimizer.optimizer.param_groups[0]['lr'], self.total_iters)
				self.total_iters += 1

			self.history['train_loss'].append(train_loss_epoch/(t+1))

			if self.verbose>0:
				print('Total train loss: {:0.4f}'.format(self.history['train_loss'][-1]))

			if self.valid_loader is not None:

				scores, labels = None, None

				for t, batch in enumerate(self.valid_loader):
					scores_batch, labels_batch = self.valid(batch)

					try:
						scores = np.concatenate([scores, scores_batch], 0)
						labels = np.concatenate([labels, labels_batch], 0)
					except:
						scores, labels = scores_batch, labels_batch

				self.history['valid_loss'].append(compute_eer(labels, scores))

				if self.logger:
					self.logger.add_scalar('Valid EER', self.history['valid_loss'][-1], np.min(self.history['valid_loss']), self.total_iters)
					self.logger.add_scalar('Best valid EER', np.min(self.history['valid_loss']), self.total_iters)
					self.logger.add_pr_curve('Valid. ROC', labels=labels, predictions=scores, global_step=self.total_iters)

				if self.verbose>0:
					print('Current validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_loss'][-1], np.min(self.history['valid_loss']), 1+np.argmin(self.history['valid_loss'])))

			if self.verbose>0:
				print('Current LR: {}'.format(self.optimizer.optimizer.param_groups[0]['lr']))

			self.cur_epoch += 1

			if ( self.cur_epoch % save_every == 0 or self.history['valid_loss'][-1] < np.min([np.inf]+self.history['valid_loss'][:-1]) ) and self.save_cp:
				self.checkpointing()

		if self.verbose>0:
			print('Training done!')

		if self.valid_loader is not None:
			if self.verbose>0:
				print('Best validation loss and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['valid_loss']), 1+np.argmin(self.history['valid_loss'])))
			return np.min(self.history['valid_loss'])

	def train_step(self, batch):

		self.model.train()
		self.optimizer.zero_grad()

		fv_clean, fv_attack, y_clean, y_attack = batch

		fv = torch.cat([fv_clean, fv_attack],0)
		y = torch.cat([y_clean, y_attack],0)

		if self.cuda_mode:
			fv, y = fv.to(self.device), y.to(self.device)

		pred = self.model.forward(fv)

		loss = torch.nn.BCEWithLogitsLoss()(pred, y)

		loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gnorm)
		self.optimizer.step()

		if self.logger:
			self.logger.add_scalar('Info/Grad_norm', grad_norm, self.total_iters)

		return loss.item()

	def valid(self, batch):

		self.model.eval()

		with torch.no_grad():

			fv_clean, fv_attack, y_clean, y_attack = batch

			fv = torch.cat([fv_clean, fv_attack],0)
			y = torch.cat([y_clean, y_attack],0)

			if self.cuda_mode:
				fv, y = fv.to(self.device), y.to(self.device)

			pred = self.model.forward(fv)

		return torch.sigmoid(pred).detach().cpu().numpy().squeeze(), y.cpu().numpy().squeeze()

	def checkpointing(self):

		# Checkpointing
		if self.verbose>0:
			print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'input_size': self.model.input_size,
		'n_hidden': self.model.n_hidden,
		'hidden_size': self.model.hidden_size,
		'dropout_prob': self.model.dropout_prob,
		'optimizer_state': self.optimizer.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'cur_epoch': self.cur_epoch}
		torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt, map_location = lambda storage, loc: storage)
			# Load model state
			if self.model.input_size!=ckpt['input_size'] or self.model.n_hidden!=ckpt['n_hidden'] or self.model.hidden_size!=ckpt['hidden_size'] or self.model.dropout_prob!=ckpt['dropout_prob']:
				print('Reinstantiating model with correct configuration')
				self.model = model_.MLP(n_in=ckpt['input_size'], nh=ckpt['n_hidden'], n_h=ckpt['hidden_size'], dropout_prob=ckpt['dropout_prob'])
				print(self.model)
			self.model.load_state_dict(ckpt['model_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			if self.cuda_mode:
				self.model = self.model.to(self.device)

		else:
			print('No checkpoint found at: {}'.format(ckpt))
