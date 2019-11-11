import argparse
import numpy as np
import glob
import torch
import torch.nn.functional as F
import os
from kaldi_io import read_vec_flt_scp
import model as model_
import scipy.io as sio
from tqdm import tqdm
from utils import compute_eer_labels, get_freer_gpu, parse_data_dict

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compute scores')
	parser.add_argument('--path-to-data', type=str, default='./data/feats.scp', metavar='Path', help='Path to input data')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for file containing model')
	parser.add_argument('--out-path', type=str, default='./out.txt', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--no-output-file', action='store_true', default=False, help='Disables writing scores into out file')
	parser.add_argument('--no-eer', action='store_true', default=False, help='Disables computation of EER')
	parser.add_argument('--eval', action='store_true', default=False, help='Enables eval trials reading')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	if os.path.isfile(args.out_path):
		os.remove(args.out_path)
		print(args.out_path + ' Removed')

	print('Cuda Mode is: {}'.format(args.cuda))

	if args.cuda:
		device = get_freer_gpu()
	else:
		device = torch.device('cpu')

	print('Loading model')

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	model = model_.MLP(n_in=ckpt['input_size'], nh=ckpt['n_hidden'], n_h=ckpt['hidden_size'], dropout_prob=ckpt['dropout_prob'])
	model.load_state_dict(ckpt['model_state'], strict=True)
	model.eval()

	print('Model loaded')

	print('Loading data')

	data = { k:m for k,m in read_vec_flt_scp(args.path_to_data) }

	test_utts, label_list, data = parse_data_dict(data)

	print('Data loaded')

	print('Start of scores computation')

	score_list = []

	with torch.no_grad():

		iterator = tqdm(enumerate(test_utts), total=len(test_utts))

		for i, utt in iterator:

			feats = torch.from_numpy(data[utt]).unsqueeze(0).float()

			try:
				if args.cuda:
					feats = feats.to(device)
					model = model.to(device)

				score = 1.-torch.sigmoid(model.forward(feats)).item()

			except:
				feats = feats.cpu()
				model = model.cpu()

				score = 1.-torch.sigmoid(model.forward(feats)).item()

			score_list.append(score)

	if not args.no_output_file:

		print('Storing scores in output file:')
		print(args.out_path)

		with open(args.out_path, 'w') as f:
			if args.eval:
				for i, utt in enumerate(test_utts):
					f.write("%s" % ' '.join([utt, str(score_list[i])+'\n']))
			else:
				for i, utt in enumerate(test_utts):
					f.write("%s" % ' '.join([utt, label_list[i], str(score_list[i])+'\n']))

	if not args.no_eer and not args.eval:
		print('EER: {}'.format(compute_eer_labels(label_list, score_list)))

	print('All done!!')
