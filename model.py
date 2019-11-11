import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class MLP(nn.Module):
	def __init__(self, n_in=20000, nh=1, n_h=512, dropout_prob=0.25):
		super(MLP, self).__init__()

		self.input_size = n_in
		self.n_hidden = nh
		self.hidden_size = n_h
		self.dropout_prob = dropout_prob

		self.classifier = self.make_bin_layers(n_in=n_in, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob)

		self.initialize_params()

	def forward(self, x):

		for layer in self.classifier:
			x = layer(x)

		return x

	def make_bin_layers(self, n_in, n_h_layers, h_size, dropout_p):

		classifier = nn.ModuleList([nn.Linear(n_in, h_size), nn.LeakyReLU(0.1)])

		for i in range(n_h_layers-1):
			classifier.append(nn.Linear(h_size, h_size))
			classifier.append(nn.LeakyReLU(0.1))

		classifier.append(nn.Dropout(p=dropout_p))
		classifier.append(nn.Linear(h_size, 1))

		return classifier

	def initialize_params(self):
		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight)
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()
