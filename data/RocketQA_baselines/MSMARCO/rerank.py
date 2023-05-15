import os
import sys
from tqdm import tqdm
import rocketqa


model_name, output_folder = sys.argv[1], sys.argv[2]
model = rocketqa.load_model(model_name, use_cuda=True, device_id=0, batch_size=100)

pid2p = dict()
with open('../../raw/MSMARCO/collection.tsv', 'r') as f:
    next(f)
    for line in tqdm(f, desc='loading passages'):
        pid, text = line.strip().split('\t')
        pid2p[pid] = text

for split in ['train', 'dev']:
    if split == 'train':
        queries_file = '../../raw/MSMARCO/queries.train.tsv'
    else:
        queries_file = '../../raw/MSMARCO/queries.dev.small.tsv'

    qid2q = dict()
    with open(queries_file, 'r') as f:
        for line in f:
            qid, q = line.strip().split('\t')
            qid2q[qid] = q

    qid2pids = dict()
    with open(f'{output_folder}/res.top100.' + split, 'r') as f:
        for line in tqdm(f, desc='loading ' + split + ' retrieval results'):
            qid, pid, rank, score = line.strip().split()
            if qid in qid2pids:
                qid2pids[qid].append(pid)
            else:
                qid2pids[qid] = [pid]
    
    with open(f'{output_folder}/rerank.res.top100.' + split, 'w') as f:
        for qid, pids in tqdm(qid2pids.items(), desc='reranking'):
            if len(pids) != 100:
                pids = pids[:100]
                # print(f'retrieval list for question {qid} truncated.')
            
            query = [qid2q[qid]] * 100
            para = [pid2p[pid] for pid in pids]
            
            scores = list(model.matching(query=query, para=para))
            
            sorted_index = [idx for idx, x in sorted(list(enumerate(scores)), key=lambda x:x[1], reverse=True)]
            sorted_pids = [pids[idx] for idx in sorted_index]
            sorted_scores = [scores[idx] for idx in sorted_index]
            for rank, (pid, score) in enumerate(zip(sorted_pids, sorted_scores)):
                f.write('%s\t%s\t%d\t%s\n' % (qid, pid, rank, score))

