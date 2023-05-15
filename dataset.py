import os
import h5py
import pickle
import logging
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

logger = logging.getLogger()


class RetrievalSim(Dataset):
    r"""
    :returns
        sim_matrix: FloatTensor with size [C, 1 + list_len, num_anchors]
        label: BoolTensor with size [list_len], relevant or not
    """

    def __init__(self, root='data', split='train', list_len=100, num_anchors=100,
                 shuffle_retrieval_list=False, perm_feat_dim=False):
        self.root = root
        self.split = split
        self.list_len = int(list_len)
        self.num_anchors = int(num_anchors)
        self.shuffle_retrieval_list = shuffle_retrieval_list
        self.perm_feat_dim = perm_feat_dim

        logger.info(f'Loading {split} dataset...')
        with open(os.path.join(root, split + '_ground_truth.pkl'), 'rb') as f:
            self.data = pickle.load(f)

        logger.info('Preparing features...')
        self.prepare_features()

        logger.info('Preparing labels...')
        self.prepare_labels()

        self.filter_data()

        self.qs = sorted(list(self.data.keys()))  # sort to guarantee reproducibility
        logger.info(f'Total number of questions: {len(self.qs)}')

        pos_num = []
        for q in self.data:
            pos_num.append(self.data[q]['label'].sum().item())
        logger.info(f'Average positive number: {sum(pos_num) / len(pos_num)}')

    def filter_data(self):
        # remove samples without positive in training
        if self.split == 'train':
            unlabeled_qs = [q for q in self.data if self.data[q]['label'].sum() == 0]
            for q in unlabeled_qs:
                self.data.pop(q)

    def prepare_features(self):
        with h5py.File(os.path.join(self.root, self.split + '_scores.hdf5'), 'r') as f:
            for q in tqdm(self.data):
                bm25_matrix = torch.FloatTensor(np.concatenate((f[q]['query_ctx_bm25_score'][()],
                                                                f[q]['ctx_ctx_bm25_score'][()]), axis=0))
                bm25_matrix = bm25_matrix[:int(1 + self.list_len), :int(self.num_anchors)]
                bm25_matrix = self.norm_feature(bm25_matrix, 100)
                dense_matrix = torch.FloatTensor(np.concatenate((f[q]['query_ctx_dense_score'][()],
                                                                 f[q]['ctx_ctx_dense_score'][()]), axis=0))
                dense_matrix = dense_matrix[:int(1 + self.list_len), :int(self.num_anchors)]
                dense_matrix = self.norm_feature(dense_matrix, 10)
                self.data[q]['feature'] = torch.stack([bm25_matrix, dense_matrix], dim=0)

    @staticmethod
    def norm_feature(x, norm_temperature=None):
        if norm_temperature is not None:
            x = (x / norm_temperature).softmax(dim=-1)

        # max-min normalization
        norm_min = x.min(dim=-1, keepdim=True).values
        norm_max = x.max(dim=-1, keepdim=True).values
        x = (x - norm_min) / (norm_max - norm_min + 1e-10)
        x = x * 2.0 - 1
        return x

    def prepare_labels(self):
        for q, d in self.data.items():
            if 'has_answer' in d and self.split != 'train':
                # for NQ evaluation
                has_answer = d['has_answer']
                label = [hit for hit in has_answer]
            else:
                positive_ctxs = d['positive_ctxs']
                retrieved_ctxs = d['retrieved_ctxs']
                label = [pid in positive_ctxs for pid in retrieved_ctxs]
            self.data[q]['label'] = torch.BoolTensor(label)[:int(self.list_len)]

    def __getitem__(self, index):
        q = self.qs[index]
        sim_matrix = self.data[q]['feature']
        label = self.data[q]['label']

        if self.shuffle_retrieval_list:
            label_perm_idx = torch.randperm(label.shape[0])
            label = label[label_perm_idx]

            matrix_perm_idx = torch.cat((torch.zeros(1, dtype=torch.long),
                                         label_perm_idx + torch.scalar_tensor(1, dtype=torch.long)))
            sim_matrix = sim_matrix[matrix_perm_idx]
            if self.perm_feat_dim:
                sim_matrix = sim_matrix[:, matrix_perm_idx]

        return q, sim_matrix, label

    def __len__(self):
        return len(self.qs)
