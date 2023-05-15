import os
import pickle
import h5py
import argparse
from tqdm import tqdm
from pyserini.search import FaissSearcher
from pyserini.search.faiss.__main__ import init_query_encoder
from utils import BM25SimComputerCached, DenseSimComputer

"""
******************************************************************
ground_truth.pkl
{
    "qid":
    {
        "question": raw text of question
        "positive_ctxs": [pid1, pid2, pid3, ...], string list, variable length
        "retrieved_ctxs": [pid1, pid2, pid3, ...], string list, length of K
    }
}
******************************************************************
score.hdf5
{
    "qid":
    {
        "query_ctx_dense_score": float array, size of [1, K]
        "ctx_ctx_dense_score": float array, size of [K, K]
        "query_ctx_bm25_score": float array, size of [1, K]
        "ctx_ctx_bm25_score": float array, size of [K, K]
    }
}
******************************************************************
"""


def generate_positive_list(input_path, output_path):
    # qid, question and positive_ctxs
    for split in ('train', 'dev'):
        data = dict()
        qid2q = dict()
        file_name = 'queries.train.tsv' if split == 'train' else 'queries.dev.small.tsv'
        with open(os.path.join(input_path, file_name), 'r') as f:
            for line in f:
                qid, q = line.rstrip().split('\t')
                qid2q[qid] = q

        file_name = 'qrels.train.tsv' if split == 'train' else 'qrels.dev.small.tsv'
        with open(os.path.join(input_path, file_name), 'r') as f:
            for line in tqdm(f, desc='generating relevance list'):
                qid, _, pid, rel = line.rstrip().split('\t')
                assert rel == '1'
                question = qid2q[qid]
                if qid not in data:
                    data[qid] = {'question': question, 'positive_ctxs': []}
                data[qid]['positive_ctxs'].append(pid)

        with open(os.path.join(output_path, split + '_ground_truth.pkl'), 'wb') as f:
            pickle.dump(data, f)


def generate_retrieval_list(index, query_encoder, output_path, results_path_prefix=None):
    # retrieved_ctxs
    query_encoder = init_query_encoder(query_encoder, None, None, None, None, 'cpu', None)
    searcher = FaissSearcher.from_prebuilt_index(index, query_encoder)

    for split in ('train', 'dev'):
        with open(os.path.join(output_path, split + '_ground_truth.pkl'), 'rb') as f:
            data = pickle.load(f)

        if results_path_prefix is not None:
            with open(results_path_prefix + split, 'r') as f:
                for i, line in enumerate(tqdm(f, desc='generating retrieval list')):
                    qid, pid, rank, score = line.rstrip().split('\t')
                    assert qid in data
                    if 'retrieved_ctxs' in data[qid]:
                        data[qid]['retrieved_ctxs'].append(pid)
                    else:
                        data[qid]['retrieved_ctxs'] = [pid]
        else:
            batch_size, threads = 36, 30
            qids = list(data.keys())
            qs = [data[qid]['question'] for qid in qids]
            batch_qids, batch_qs = list(), list()
            for index, (qid, q) in enumerate(tqdm(zip(qids, qs), desc='generating retrieval list')):
                batch_qids.append(qid)
                batch_qs.append(q)
                if (index + 1) % batch_size == 0 or index == len(qids) - 1:
                    batch_hits = searcher.batch_search(batch_qs, batch_qids, k=100, threads=threads)
                    assert len(batch_qids) == len(batch_hits)

                    for qid_in_batch, hits in batch_hits.items():
                        data[qid_in_batch]['retrieved_ctxs'] = [hit.docid for hit in hits]

                    batch_qids.clear()
                    batch_qs.clear()
                else:
                    continue

        with open(os.path.join(output_path, split + '_ground_truth.pkl'), 'wb') as f:
            pickle.dump(data, f)


def generate_dense_score(index, query_encoder, output_path):
    # query_ctx_dense_score and ctx_ctx_dense_score
    query_encoder = init_query_encoder(query_encoder, None, None, None, None, 'cpu', None)
    simcomputer = DenseSimComputer.from_prebuilt_index(index, query_encoder)

    for split in ('train', 'dev'):
        with open(os.path.join(output_path, split + '_ground_truth.pkl'), 'rb') as f:
            data = pickle.load(f)

        with h5py.File(os.path.join(output_path, split + '_scores.hdf5'), 'a') as f:
            for qid in tqdm(data, desc='generating dense scores'):
                question = data[qid]['question']
                retrieved_pids = data[qid]['retrieved_ctxs']
                query_ctx_dense_score, ctx_ctx_dense_score = simcomputer.compute(question, retrieved_pids)

                if qid not in f.keys():
                    f.create_group(qid)
                f[qid]['query_ctx_dense_score'] = query_ctx_dense_score.astype(float)
                f[qid]['ctx_ctx_dense_score'] = ctx_ctx_dense_score.astype(float)


def generate_bm25_score(lucene_index_path, output_path):
    # query_ctx_bm25score and ctx_ctx_bm25score
    simcomputer = BM25SimComputerCached(lucene_index_path)

    for split in ('train', 'dev'):
        with open(os.path.join(output_path, split + '_ground_truth.pkl'), 'rb') as f:
            data = pickle.load(f)

        with h5py.File(os.path.join(output_path, split + '_scores.hdf5'), 'a') as f:
            for qid in tqdm(data, desc='generating BM25 scores'):
                question = data[qid]['question']
                retrieved_pids = data[qid]['retrieved_ctxs']
                query_ctx_bm25_score, ctx_ctx_bm25_score = simcomputer.compute(question, retrieved_pids)

                if qid not in f.keys():
                    f.create_group(qid)
                f[qid]['query_ctx_bm25_score'] = query_ctx_bm25_score.astype(float)
                f[qid]['ctx_ctx_bm25_score'] = ctx_ctx_bm25_score.astype(float)


retriever_dense_index_map = {
    'ANCE': 'msmarco-passage-ance-bf',
    'DistilBERT-KD': 'msmarco-passage-distilbert-dot-margin_mse-T2-bf',
    'TAS-B': 'msmarco-passage-distilbert-dot-tas_b-b256-bf',
    'TCT-ColBERT-v1': 'msmarco-passage-tct_colbert-bf',
    'TCT-ColBERT-v2': 'msmarco-passage-tct_colbert-v2-hnp-bf',
    'RocketQA-retriever': 'msmarco-passage-ance-bf',
    'RocketQAv2-retriever': 'msmarco-passage-ance-bf',
    'RocketQA-reranker': 'msmarco-passage-ance-bf',
    'RocketQAv2-reranker': 'msmarco-passage-ance-bf',
}

retriever_query_encoder_map = {
    'ANCE': 'castorini/ance-msmarco-passage',
    'DistilBERT-KD': 'sebastian-hofstaetter/distilbert-dot-margin_mse-T2-msmarco',
    'TAS-B': 'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco',
    'TCT-ColBERT-v1': 'castorini/tct_colbert-msmarco',
    'TCT-ColBERT-v2': 'castorini/tct_colbert-v2-hnp-msmarco',
    'RocketQA-retriever': 'castorini/ance-msmarco-passage',
    'RocketQAv2-retriever': 'castorini/ance-msmarco-passage',
    'RocketQA-reranker': 'castorini/ance-msmarco-passage',
    'RocketQAv2-reranker': 'castorini/ance-msmarco-passage',
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate MSMARCO dataset.')
    parser.add_argument('--msmarco_data_path', type=str, required=False, default='raw/MSMARCO',
                        help='Original MSMARCO retriever dataset path.')
    parser.add_argument('--retriever', type=str, required=True,
                        choices=["ANCE", "DistilBERT-KD", "TAS-B", "TCT-ColBERT-v1", "TCT-ColBERT-v2",
                                 "RocketQA-retriever", "RocketQAv2-retriever", "RocketQA-reranker", "RocketQAv2-reranker"],
                        default=None, help='Which retriever method to use.')
    parser.add_argument('--lucene_index_path', type=str, required=False,
                        default='pyserini/indexes/lucene-index-msmarco-passage',
                        help='Lucene index path.')
    parser.add_argument('--output_path', type=str, required=False, default=None,
                        help='Generated dataset path.')
    args = parser.parse_args()

    msmarco_data_path = args.msmarco_data_path
    dense_index_name = retriever_dense_index_map[args.retriever]
    query_encoder = retriever_query_encoder_map[args.retriever]
    lucene_index_path = args.lucene_index_path
    if args.output_path is None:
        output_path = 'MSMARCO_' + args.retriever
    else:
        output_path = args.output_path

    if 'RocketQA' in args.retriever:
        results_path_prefix = f'RocketQA_baselines/MSMARCO/{args.retriever.split("-")[0]}/res.top100.'
        if 'reranker' in args.retriever:
            results_path = results_path_prefix.replace('res.top100', 'rerank.res.top100')
    else:
        results_path_prefix = None

    os.makedirs(output_path, exist_ok=True)
    generate_positive_list(msmarco_data_path, output_path)
    generate_retrieval_list(dense_index_name, query_encoder, output_path, results_path_prefix)  # 5.5h
    generate_dense_score(dense_index_name, query_encoder, output_path)  # 2h
    generate_bm25_score(lucene_index_path, output_path)  # 15h
