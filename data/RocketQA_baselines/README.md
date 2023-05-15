## Install RocketQA

```shell
conda create -n rocketqa python==3.9.12
conda activate rocketqa
conda install paddlepaddle-gpu==2.4.2 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
pip install rocketqa scipy faiss-cpu
```

## Retrieval and Rerank NQ
```shell
bash NQ/retrieval_rerank.sh
```

## Retrieval and Rerank TRECDL
```shell
bash MSMARCO/retrieval_rerank.sh
```

## Retrieval and Rerank TRECDL
```shell
bash TRECDL/retrieval_rerank.sh
```