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

    for split in ['2019', '2020']:
        qrels_file = f'../../raw/TRECDL/{split}qrels-pass.txt'
        queries_file = f'../../raw/TRECDL/msmarco-test{split}-queries.tsv'

        qid2q = dict()
        with open(queries_file, 'r') as f:
            for line in f:
                qid, q = line.strip().split('\t')
                qid2q[qid] = q
        
        results = []

        with open(qrels_file, 'r') as f:
            lines = [line.strip().split() for line in f.readlines()]
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
    logging.info("Passage already encoded in MSMARCO.")