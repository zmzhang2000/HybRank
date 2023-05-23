import os
import sys
from tqdm import tqdm
import rocketqa


if sys.argv[1] == 'v1_nq_ce':
    model_name = 'v1_nq_ce'
elif sys.argv[1] == 'v2_nq_ce':
    model_name = 'v2_nq_ce/config.json'
else:
    raise NotImplementedError
output_folder = sys.argv[2]
model = rocketqa.load_model(model_name, use_cuda=True, device_id=0, batch_size=100)

pid2p = dict()
with open('../../raw/NQ/data/wikipedia_split/psgs_w100.tsv', 'r') as f:
    for line in tqdm(f, desc='loading passages'):
        pid, text, title = line.strip().split('\t')
        pid2p[pid] = text[1:-1]

for split in ['train', 'dev', 'test']:
    qid2q = dict()
    with open(os.path.join('../../raw/NQ/data/retriever/qas', 'nq-' + split + '.csv'), 'r') as f:
        for qid, line in enumerate(tqdm(f, desc='loading ' + split + ' queries')):
            query, answers = line.strip().split('\t')
            qid2q[str(qid)] = query

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
            assert len(pids) == 100
            
            query = [qid2q[qid]] * 100
            para = [pid2p[pid] for pid in pids]
            
            scores = list(model.matching(query=query, para=para))
            
            sorted_index = [idx for idx, x in sorted(list(enumerate(scores)), key=lambda x:x[1], reverse=True)]
            sorted_pids = [pids[idx] for idx in sorted_index]
            sorted_scores = [scores[idx] for idx in sorted_index]
            for rank, (pid, score) in enumerate(zip(sorted_pids, sorted_scores)):
                f.write('%s\t%s\t%d\t%s\n' % (qid, pid, rank, score))

