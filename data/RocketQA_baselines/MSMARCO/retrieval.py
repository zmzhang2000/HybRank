import os
import sys
import glob
import pickle
import logging
import numpy as np
import faiss


logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

topk = 100
model_name, output_folder = sys.argv[1], sys.argv[2]
os.makedirs(output_folder, exist_ok=True)
index_path = f'{output_folder}/index'
index_meta_path = f'{output_folder}/index_meta'
q_embed_pattern = f'{output_folder}/q_embed_*.pkl'
p_embed_pattern = f'{output_folder}/p_embed_*.pkl'
output_path = f'{output_folder}/res.top100'

if os.path.exists(index_path) and os.path.exists(index_meta_path):
    logging.info('reading index ' + index_path)
    with open(index_meta_path, 'rb') as f:
        ids = pickle.load(f)
    index = faiss.read_index(index_path)
    assert len(ids) == index.ntotal

else:
    ids = []
    index = faiss.IndexFlatIP(768)
    for p_embed_file in glob.glob(p_embed_pattern):
        logging.info('indexing ' + p_embed_file)
        with open(p_embed_file, "rb") as f:
            data = pickle.load(f)
        buffer = []
        buffer_size = 50000
        for db_id, doc_vector in data:
            buffer.append((db_id, doc_vector))
            if buffer_size == len(buffer):
                ids.extend([x[0] for x in buffer])
                index.add(np.concatenate([np.reshape(t[1], (1, -1)) for t in buffer], axis=0))
                buffer = []
        if len(buffer) > 0:
            ids.extend([x[0] for x in buffer])
            index.add(np.concatenate([np.reshape(t[1], (1, -1)) for t in buffer], axis=0))
            buffer = []
        logging.info('done. ntotal = ' + str(index.ntotal))
    
    faiss.write_index(index, index_path)
    with open(index_meta_path, mode='wb') as f:
        pickle.dump(ids, f)

logging.info("indexing done.")

buffer_size = 32
faiss.omp_set_num_threads(30)
for split in ['train', 'dev']:
    with open(q_embed_pattern.replace('*', split), 'rb') as f:
        data = pickle.load(f)
    
    new_output_path = output_path + '.' + split
    if os.path.exists(new_output_path):
        os.remove(new_output_path)
    for start_idx in range(0, len(data), buffer_size):
        qids = [x[0] for x in data[start_idx:(start_idx + buffer_size)]]
        qvecs = np.stack([x[1] for x in data[start_idx:(start_idx + buffer_size)]], axis=0)
        scores, indexes = index.search(qvecs, topk)
        retrieved_lists = [[ids[i] for i in query_top_idxs] for query_top_idxs in indexes]
        assert len(scores) == len(retrieved_lists) == len(qids)

        with open(new_output_path, 'a') as f:
            for qid, pid_list, score_list in zip(qids, retrieved_lists, scores):
                for rank, (pid, score) in enumerate(zip(pid_list, score_list), 1):
                    f.write("%s\t%s\t%d\t%f\n" % (qid, pid, rank, score))
        
        logging.info(f"{start_idx} queries searched.")

