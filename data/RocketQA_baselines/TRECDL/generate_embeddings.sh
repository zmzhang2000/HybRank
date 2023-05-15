#!/bin/bash
set -x

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

wait