#!/usr/local/bin bash

name=libritrans-en-fr
km_model=libritrans_en_fr-train-0.01-hubert-6-50-simple

hidden_input_dir=data/s2u/$name/$km_model
tsv_input_dir=data/s2t/$name/normalized
audio_key=audio
output_dir=data/u2t/$name/$km_model
ext=wav

mkdir -p $output_dir

for split in test dev train; do
    python convert/u2t.py \
        --hidden-unit-file $hidden_input_dir/$split.txt \
        --input-tsv $tsv_input_dir/$split.tsv \
        --output-tsv $output_dir/$split.tsv \
        --key $audio_key \
        --extension $ext
done
