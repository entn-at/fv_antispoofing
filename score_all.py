import argparse
import numpy as np
import glob
import torch
import torch.nn.functional as F
import os
from kaldi_io import read_vec_flt_scp
import model as model_
import scipy.io as sio
import glob
from tqdm import tqdm
from utils import compute_eer_labels, get_freer_gpu, read_trials

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compute scores')
	parser.add_argument('--path-to-data', type=str, default='./data/feats.scp', metavar='Path', help='Path to input data')
	parser.add_argument('--trials-path', type=str, default='./data/trials', metavar='Path', help='Path to trials file')
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

	print('Loading data')

	data = { k.split('-')[0]:m for k,m in read_vec_flt_scp(args.path_to_data) }

	if args.eval:
		test_utts = read_trials(args.trials_path, eval_=args.eval)
	else:
		test_utts, attack_type_list, label_list = read_trials(args.trials_path, eval_=args.eval)

	print('Data loaded')

	cp_list = glob.glob(args.cp_path+'*.pt')

	print('Models to evaluate: {}'.format(len(cp_list)))

	print('Cuda Mode: {}'.format(args.cuda))

	if args.cuda:
		device = get_freer_gpu()
	else:
		device = torch.device('cpu')

	scores_dict = {}
	perf_dict = {}
	pef_list = []

	for ind_model in tqdm(cp_list, total=len(cp_list)):

		try:
			ckpt = torch.load(ind_model, map_location = lambda storage, loc: storage)
			model = model_.MLP(n_in=ckpt['input_size'], nh=ckpt['n_hidden'], n_h=ckpt['hidden_size'], dropout_prob=ckpt['dropout_prob'])
			model.load_state_dict(ckpt['model_state'], strict=True)
			model.eval()

			score_list = []

			with torch.no_grad():

				for i, utt in enumerate(test_utts):

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

				model_id = ind_model.split(args.cp_path)[-1]
				perf_list.append(compute_eer_labels(label_list, score_list))
				scores_dict[model_id] = score_list
				perf_dict[model_id] = perf_list[-1]


		except RuntimeError as err:
			print('\nRuntime Error: {0}'.format(err))
			print('\nSkipping model {}\n'.format(ind_model))

	print('\nScoring done\n')

	print('Avg: {}'.format(np.mean(pef_list)))
	print('Std: {}'.format(np.std(pef_list)))
	print('Median: {}'.format(np.median(pef_list)))
	print('Max: {}'.format(np.max(pef_list)))
	print('Min: {}'.format(np.min(pef_list)))

	final_scores = []
	perf_target = np.median(pef_list)

	for model_id in perf_dict:
		if perf_dict[perf_dict] <= perf_target:
			final_scores.append(scores_dict[model_id])

	final_scores=np.mean(final_scores, 0)

	if not args.no_output_file:

		print('\nStoring scores in output file:')
		print(args.out_path)

		with open(args.out_path, 'w') as f:
			if args.eval:
				for i, utt in enumerate(test_utts):
					f.write("%s" % ' '.join([utt, str(final_scores[i])+'\n']))
			else:
				for i, utt in enumerate(test_utts):
					f.write("%s" % ' '.join([utt, attack_type_list[i], label_list[i], str(final_scores[i])+'\n']))

	if not args.no_eer and not args.eval:
		print('\nFinal EER: {}'.format(compute_eer_labels(label_list, final_scores)))

	print('\nAll done!!')
