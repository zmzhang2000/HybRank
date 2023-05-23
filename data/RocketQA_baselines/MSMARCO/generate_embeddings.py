import os
import sys
import logging
import pickle
import rocketqa


logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

shard_id = 0 if sys.argv[1] == 'q' else int(sys.argv[1])
shard_num, device_id, batch_size = [int(x) for x in sys.argv[2:5]]
model_name, output_folder = sys.argv[5:7]

if sys.argv[1] == 'q':
    model = rocketqa.load_model(model_name, use_cuda=True, device_id=device_id, batch_size=batch_size)

    for split in ['train', 'dev']:
        if split == 'train':
            qrels_file = '../../raw/MSMARCO/qrels.train.tsv'
            queries_file = '../../raw/MSMARCO/queries.train.tsv'
        else:
            qrels_file = '../../raw/MSMARCO/qrels.dev.small.tsv'
            queries_file = '../../raw/MSMARCO/queries.dev.small.tsv'

        qid2q = dict()
        with open(queries_file, 'r') as f:
            for line in f:
                qid, q = line.strip().split('\t')
                qid2q[qid] = q
        
        results = []

        with open(qrels_file, 'r') as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            qids = sorted(list(set([x[0] for x in lines])))
            texts = [qid2q[qid] for qid in qids]
            q_num = len(qids)
            batch_qids, batch_texts = [], []
            for idx, (qid, text) in enumerate(zip(qids, texts)):                
                batch_qids.append(qid)
                batch_texts.append(text)
                if len(batch_texts) == batch_size or idx == q_num - 1:
                    batch_outs = list(model.encode_query(query=batch_texts))
                    assert len(batch_outs) == len(batch_qids)
                    results.extend([(qid, out) for qid, out in zip(batch_qids, batch_outs)])
                    batch_qids, batch_texts = [], []
                    logging.info("%d query encoded." % idx)

        os.makedirs(output_folder, exist_ok=True)
        with open(f'{output_folder}/q_embed_' + split + '.pkl', 'wb') as f:
            pickle.dump(results, f)

else:
    logging.info((shard_id, shard_num, device_id, batch_size))
    assert shard_id < shard_num

    model = rocketqa.load_model(model_name, use_cuda=True, device_id=device_id, batch_size=batch_size)

    corpus_file = '../../raw/MSMARCO/collection.tsv'

    with open(corpus_file) as f:
        para_num = len([idx for idx, _ in enumerate(f)])
    shard_size = para_num // shard_num
    start_idx = shard_id * shard_size
    end_idx = para_num + 1 if shard_id == shard_num - 1 else start_idx + shard_size

    results = []

    with open(corpus_file, 'r') as f:
        next(f)
        batch_pids, batch_texts = [], []
        for idx, line in enumerate(f):
            if idx >= start_idx and idx < end_idx:
                pid, text = line.strip().split('\t')
                
                batch_pids.append(pid)
                batch_texts.append(text)
                if len(batch_texts) == batch_size or idx == end_idx - 1:
                    logging.info("encoding passage %s to %s ..." % (batch_pids[0], batch_pids[-1]))
                    batch_outs = list(model.encode_para(para=batch_texts))
                    assert len(batch_outs) == len(batch_pids)
                    results.extend([(pid, out) for pid, out in zip(batch_pids, batch_outs)])
                    batch_pids, batch_texts = [], []

    os.makedirs(output_folder, exist_ok=True)
    with open(f'{output_folder}/p_embed_' + str(shard_id) + '.pkl', 'wb') as f:
        pickle.dump(results, f)