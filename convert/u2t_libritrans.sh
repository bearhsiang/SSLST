#!/usr/local/bin bash

name=libritrans-en-fr
feature_type=logmel
n_cluster=50
mode=simple

hidden_input_dir=data/s2u/$name/quantized-$feature_type-$n_cluster-$mode
tsv_input_dir=data/s2t/$name/normalized
index_key=id
output_dir=data/u2t/$name/$feature_type-$n_cluster-$mode

mkdir -p $output_dir

for split in test dev train; do
    python convert/u2t.py \
        --hidden-unit-file $hidden_input_dir/$split.txt \
        --input-tsv $tsv_input_dir/$split.tsv \
        --output-tsv $output_dir/$split.tsv \
        --key $index_key
done
