# Prepare Data


## Dependencies
```
java 11.0.16
maven 3.8.6
anserini 0.14.3
faiss-cpu 1.7.2
pyserini 0.17.1
```


## Natural Questions

### 1. Download raw data (Refer to [DPR](https://github.com/facebookresearch/DPR) for more details of the dataset)
```shell
python download_DPR_data.py --resource data.wikipedia_split.psgs_w100
python download_DPR_data.py --resource data.retriever.nq
python download_DPR_data.py --resource data.retriever.qas.nq
mkdir -p raw && mv downloads raw/NQ
```

### 2. Convert collections to jsonl format for Pyserini
```shell
python convert_NQ_collection_to_jsonl.py --collection-path raw/NQ/data/wikipedia_split/psgs_w100.tsv --output-folder pyserini/collections/NQ
```

### 3. Build Lucene indexes via Pyserini
```shell
python -m pyserini.index.lucene \
--collection JsonCollection \
--input pyserini/collections/NQ \
--index pyserini/indexes/NQ \
--generator DefaultLuceneDocumentGenerator \
--threads 1 \
--storePositions --storeDocvectors --storeRaw
```

### 4. Generate data
```shell
RETRIEVERS=("DPR-Multi" "DPR-Single" "ANCE" "FiD-KD" "RocketQA-retriever" "RocketQAv2-retriever" "RocketQA-reranker" "RocketQAv2-reranker")

for RETRIEVER in ${RETRIEVERS[@]}; do
  python generate_NQ_data.py --retriever $RETRIEVER
done
```
Note that before generate data for retriever `RocketQA*`, please generate the retrieval results following the instructions in `data/RocketQA_baselines/README.md`. Data for other retrievers can be generated directly.


## MS MARCO & TREC 2019/2020

### 1. Download raw data (Refer to [MS MARCO](https://microsoft.github.io/msmarco/) for more details of the dataset)
  * Download and uncompress MSMARCO Passage Ranking Collections and Queries [collectionandqueries.tar.gz](https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz) to `data/raw/MSMARCO/`
  * TREC DL Test Queries and Qrels
    * [TREC DL-2019](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html)
      * Download and uncompress [msmarco-test2019-queries.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz)
      * Download [2019qrels-pass.txt](https://trec.nist.gov/data/deep/2019qrels-pass.txt)
    * [TREC DL-2020](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020)
      * Download and uncompress [msmarco-test2019-queries.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz)
      * Download [2019qrels-pass.txt](https://trec.nist.gov/data/deep/2020qrels-pass.txt)
    * Put them into `data/raw/TRECDL/`

### 2. Convert collections to jsonl format for Pyserini

```shell
python convert_MSMARCO_collection_to_jsonl.py --collection-path raw/MSMARCO/collection.tsv --output-folder pyserini/collections/MSMARCO
```

### 3. Build Lucene indexes via Pyserini

```shell
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input pyserini/collections/MSMARCO \
  --index pyserini/indexes/MSMARCO \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
```

### 4. Generate data
```shell
RETRIEVERS=("ANCE" "DistilBERT-KD" "TAS-B" "TCT-ColBERT-v1" "TCT-ColBERT-v2" "RocketQA-retriever" "RocketQAv2-retriever" "RocketQA-reranker" "RocketQAv2-reranker")

for RETRIEVER in ${RETRIEVERS[@]}; do
  python generate_MSMARCO_data.py --retriever $RETRIEVER
done
```

```shell
RETRIEVERS=("ANCE" "DistilBERT-KD" "TAS-B" "TCT-ColBERT-v1" "TCT-ColBERT-v2" "RocketQA-retriever" "RocketQAv2-retriever" "RocketQA-reranker" "RocketQAv2-reranker")

SPLITS=("2019" "2020")

for RETRIEVER in ${RETRIEVERS[@]}; do
  for SPLIT in ${SPLITS[@]}; do
    python generate_TRECDL_data.py --split $SPLIT --retriever $RETRIEVER
  done
done
```