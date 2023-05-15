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

    for split in ['train', 'dev', 'test']:
        query_file = '../../raw/NQ/data/retriever/qas/nq-' + split + '.csv'
        results = []

        with open(query_file, 'r') as f:
            batch_qids, batch_texts = [], []
            lines = f.readlines()
            line_num = len(lines)
            for idx, line in enumerate(lines):
                query, answers = line.strip().split('\t')
                
                batch_qids.append(str(idx))
                batch_texts.append(query)
                if len(batch_texts) == batch_size or idx == line_num - 1:
                    logging.info("encoding query %s to %s ..." % (batch_qids[0], batch_qids[-1]))
                    batch_outs = list(model.encode_query(query=batch_texts))
                    assert len(batch_outs) == len(batch_qids)
                    results.extend([(qid, out) for qid, out in zip(batch_qids, batch_outs)])
                    batch_qids, batch_texts = [], []

        os.makedirs(output_folder, exist_ok=True)
        with open(f'{output_folder}/q_embed_' + split + '.pkl', 'wb') as f:
            pickle.dump(results, f)

else:
    logging.info((shard_id, shard_num, device_id, batch_size))
    assert shard_id < shard_num

    model = rocketqa.load_model(model_name, use_cuda=True, device_id=device_id, batch_size=batch_size)

    para_num = 21015324
    shard_size = para_num // shard_num
    start_idx = shard_id * shard_size
    end_idx = para_num + 1 if shard_id == shard_num - 1 else start_idx + shard_size

    corpus_file = '../../raw/NQ/data/wikipedia_split/psgs_w100.tsv'

    results = []

    with open(corpus_file, 'r') as f:
        next(f)
        batch_pids, batch_texts, batch_titles = [], [], []
        for idx, line in enumerate(f):
            if idx >= start_idx and idx < end_idx:
                pid, text, title = line.strip().split('\t')
                text = text[1:-1]
                
                batch_pids.append(pid)
                batch_texts.append(text)
                batch_titles.append(title)
                if len(batch_texts) == batch_size or idx == end_idx - 1:
                    logging.info("encoding passage %s to %s ..." % (batch_pids[0], batch_pids[-1]))
                    batch_outs = list(model.encode_para(para=batch_texts, title=batch_titles))
                    assert len(batch_outs) == len(batch_pids)
                    results.extend([(pid, out) for pid, out in zip(batch_pids, batch_outs)])
                    batch_pids, batch_texts, batch_titles = [], [], []

    os.makedirs(output_folder, exist_ok=True)
    with open(f'{output_folder}/p_embed_' + str(shard_id) + '.pkl', 'wb') as f:
        pickle.dump(results, f)