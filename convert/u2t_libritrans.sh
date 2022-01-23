#!/usr/local/bin bash

name=libritrans-en-fr
feature_type=hubert
n_cluster=50
layer=6
mode=simple
ext=wav

hidden_input_dir=data/s2u/$name/$feature_type-$n_cluster-$layer-$mode
tsv_input_dir=data/s2t/$name/normalized
audio_key=audio
output_dir=data/u2t/$name/$feature_type-$n_cluster-$mode-$layer

mkdir -p $output_dir

for split in test dev train; do
    python convert/u2t.py \
        --hidden-unit-file $hidden_input_dir/$split.txt \
        --input-tsv $tsv_input_dir/$split.tsv \
        --output-tsv $output_dir/$split.tsv \
        --key $audio_key \
        --extension $ext
done
