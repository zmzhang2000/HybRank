import os
import json
import pickle
import h5py
import argparse
from tqdm import tqdm
from pyserini.search import FaissSearcher
from pyserini.search.faiss.__main__ import init_query_encoder
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from utils import BM25SimComputerCached, DenseSimComputer

"""
******************************************************************
ground_truth.pkl
{
    "qid":
    {
        "question": raw text of question
        "answers": [answer1, answer2, ...], string list, variable length
        "positive_ctxs": [pid1, pid2, pid3, ...], string list, variable length
        "retrieved_ctxs": [pid1, pid2, pid3, ...], string list, length of K
        "has_answer": [True, False, True, ...], bool list, length of K
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
    # qid, question, answers and positive_ctxs
    for split in ('train', 'dev', 'test'):
        data = dict()
        q2id = dict()
        with open(os.path.join(input_path, 'qas', 'nq-' + split + '.csv'), 'r') as f:
            for i, line in enumerate(tqdm(f, desc='reading questions')):
                question, answer_str = line.rstrip().split('\t')
                data[str(i)] = {'question': question, 'answers': eval(answer_str), 'positive_ctxs': []}
                q2id[question] = str(i)

        if split != 'test':
            with open(os.path.join(input_path, 'nq-' + split + '.json'), 'r') as f:
                anno = json.load(f)
                for sample in tqdm(anno, desc='generating relevance list'):
                    qid = q2id[sample['question']]
                    data[qid]['positive_ctxs'] = [x['passage_id'] for x in sample['positive_ctxs']]

        with open(os.path.join(output_path, split + '_ground_truth.pkl'), 'wb') as f:
            pickle.dump(data, f)


def generate_retrieval_list(index, query_encoder, output_path, results_path_prefix=None):
    # retrieved_ctxs and has_answer
    query_encoder = init_query_encoder(query_encoder, None, None, None, None, 'cpu', None)
    searcher = FaissSearcher.from_prebuilt_index(index, query_encoder)
    tokenizer = SimpleTokenizer()

    for split in ('train', 'dev', 'test'):
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

        for qid in tqdm(data, desc='computing has_answer'):
            answers = data[qid]['answers']
            data[qid]['has_answer'] = []
            for docid in data[qid]['retrieved_ctxs']:
                text = json.loads(searcher.doc(docid).raw())['contents'].split('\n')[1]
                has_answer = has_answers(text, answers, tokenizer)
                data[qid]['has_answer'].append(has_answer)

        with open(os.path.join(output_path, split + '_ground_truth.pkl'), 'wb') as f:
            pickle.dump(data, f)


def generate_dense_score(index, query_encoder, output_path):
    # query_ctx_dense_score and ctx_ctx_dense_score
    query_encoder = init_query_encoder(query_encoder, None, None, None, None, 'cpu', None)
    simcomputer = DenseSimComputer.from_prebuilt_index(index, query_encoder)

    for split in ('train', 'dev', 'test'):
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

    for split in ('train', 'dev', 'test'):
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
    'DPR-Multi': 'wikipedia-dpr-multi-bf',
    'DPR-Single': 'wikipedia-dpr-single-nq-bf',
    'ANCE': 'wikipedia-ance-multi-bf',
    'FiD-KD': 'wikipedia-dpr-dkrr-nq',
    'RocketQA-retriever': 'wikipedia-dpr-dkrr-nq',
    'RocketQAv2-retriever': 'wikipedia-dpr-dkrr-nq',
    'RocketQA-reranker': 'wikipedia-dpr-dkrr-nq',
    'RocketQAv2-reranker': 'wikipedia-dpr-dkrr-nq',
}

retriever_query_encoder_map = {
    'DPR-Multi': 'facebook/dpr-question_encoder-multiset-base',
    'DPR-Single': 'facebook/dpr-question_encoder-single-nq-base',
    'ANCE': 'castorini/ance-dpr-question-multi',
    'FiD-KD': 'castorini/dkrr-dpr-nq-retriever',
    'RocketQA-retriever': 'castorini/dkrr-dpr-nq-retriever',
    'RocketQAv2-retriever': 'castorini/dkrr-dpr-nq-retriever',
    'RocketQA-reranker': 'castorini/dkrr-dpr-nq-retriever',
    'RocketQAv2-reranker': 'castorini/dkrr-dpr-nq-retriever',
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate NQ dataset.')
    parser.add_argument('--nq_data_path', type=str, required=False, default='raw/NQ/data/retriever',
                        help='Original NQ retriever dataset path.')
    parser.add_argument('--retriever', type=str, required=True,
                        choices=["DPR-Multi", "DPR-Single", "ANCE", "FiD-KD", "RocketQA-retriever", 
                                 "RocketQAv2-retriever", "RocketQA-reranker", "RocketQAv2-reranker"],
                        default=None, help='Which retriever method to use.')
    parser.add_argument('--lucene_index_path', type=str, required=False,
                        default='pyserini/indexes/wikipedia_doc',
                        help='Lucene index path.')
    parser.add_argument('--output_path', type=str, required=False, default=None,
                        help='Generated dataset path.')
    args = parser.parse_args()

    nq_data_path = args.nq_data_path
    dense_index_name = retriever_dense_index_map[args.retriever]
    query_encoder = retriever_query_encoder_map[args.retriever]
    lucene_index_path = args.lucene_index_path
    if args.output_path is None:
        output_path = 'NQ_' + args.retriever
    else:
        output_path = args.output_path

    if 'RocketQA' in args.retriever:
        results_path_prefix = f'RocketQA_baselines/NQ/{args.retriever.split("-")[0]}/res.top100.{args.split}'
        if 'reranker' in args.retriever:
            results_path_prefix = results_path_prefix.replace('res.top100', 'rerank.res.top100')
    else:
        results_path_prefix = None

    os.makedirs(output_path, exist_ok=True)
    generate_positive_list(nq_data_path, output_path)
    generate_retrieval_list(dense_index_name, query_encoder, output_path, results_path_prefix)  # 4.5h
    generate_dense_score(dense_index_name, query_encoder, output_path)  # 1h
    generate_bm25_score(lucene_index_path, output_path)  # 4.5h
