import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class Loader(Dataset):

	def __init__(self, hdf5_clean, hdf5_attack, label_smoothing=0.0, n_cycles=1):
		super(Loader, self).__init__()
		self.hdf5_1 = hdf5_clean
		self.hdf5_2 = hdf5_attack
		self.n_cycles = n_cycles

		file_1 = h5py.File(self.hdf5_1, 'r')
		self.idxlist_1 = list(file_1.keys())
		self.len_1 = len(self.idxlist_1)
		file_1.close()

		file_2 = h5py.File(self.hdf5_2, 'r')
		self.idxlist_2 = list(file_2.keys())
		self.len_2 = len(self.idxlist_2)
		file_2.close()

		self.open_file_1 = None
		self.open_file_2 = None

		self.label_smoothing = label_smoothing>0.0
		self.label_dif = label_smoothing

		print('Number of genuine and spoofing recordings: {}, {}'.format(self.len_1, self.len_2))

	def __getitem__(self, index):

		if not self.open_file_1: self.open_file_1 = h5py.File(self.hdf5_1, 'r')
		if not self.open_file_2: self.open_file_2 = h5py.File(self.hdf5_2, 'r')

		index_1 = index % self.len_1
		fv_clean = self.open_file_1[self.idxlist_1[index_1]][:]

		index_2 = index % self.len_2
		fv_attack = self.open_file_2[self.idxlist_2[index_2]][:]

		y_clean, y_attack = self.get_labels()

		return torch.from_numpy(fv_clean).float().contiguous(), torch.from_numpy(fv_attack).float().contiguous(), y_clean, y_attack

	def __len__(self):
		return self.n_cycles*np.maximum(self.len_1, self.len_2)

	def get_labels(self):

		if self.label_smoothing:
			return torch.rand(1)*self.label_dif, torch.rand(1)*self.label_dif+(1.-self.label_dif)
		else:
			return torch.zeros(1), torch.ones(1)
