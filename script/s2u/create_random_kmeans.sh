#!/usr/local/bin bash

source script/setup.sh

dim=768
n_cluster=500

mean=0
std=0.1
seed=$sslst_seed
output=$sslst_data_root/kmeans_model/random-D${dim}M${mean}S${std}-km${n_cluster}.bin

python utils_new/create_random_kmeans.py \
    -d $dim \
    -n $n_cluster \
    -m $mean \
    -s $std \
    -o $output \
    --seed $seed 