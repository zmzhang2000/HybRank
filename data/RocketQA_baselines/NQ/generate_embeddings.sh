#!/bin/bash
set -x
LD_LIBRARY_PATH=$CONDA_PREFIX/lib/
model_name=$1
output_path=$2
batch_size=512
devices=(0 1 2 3 4 5 6 7)

shard_num=${#devices[*]}

if [ ! -d $output_path ]; then
    mkdir $output_path
fi

nohup python generate_embeddings.py q 1 ${devices[0]} 256 $model_name $output_path > $output_path/q.nohup 2>&1 &
echo q start: pid=$! >> $output_path/test.log

for i in $(seq 0 $[shard_num-1]);do
    nohup python generate_embeddings.py ${i} ${shard_num} ${devices[i]} ${batch_size} $model_name $output_path > $output_path/${i}.nohup 2>&1 &
    pid[$i]=$!
    echo $i start: pid=$! >> $output_path/test.log
done

wait