set -e

cd $(dirname $0)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/

bash generate_embeddings.sh v1_nq_de RocketQA
python retrieval.py v1_nq_de RocketQA
python rerank.py v1_nq_ce RocketQA

bash wget_trained_model.sh
bash generate_embeddings.sh v2_nq_de RocketQAv2
python retrieval.py v2_nq_de RocketQAv2
python rerank.py v2_nq_ce RocketQAv2
