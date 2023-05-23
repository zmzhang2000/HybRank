set -e

cd $(dirname $0)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/

bash generate_embeddings.sh v1_marco_de RocketQA
python retrieval.py v1_marco_de RocketQA
python rerank.py v1_marco_ce RocketQA

bash generate_embeddings.sh v2_marco_de RocketQAv2
python retrieval.py v2_marco_de RocketQAv2
python rerank.py v2_marco_ce RocketQAv2
