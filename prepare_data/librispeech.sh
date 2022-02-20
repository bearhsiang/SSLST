#!/usr/local/bin bash

libri_root=/hdd/LibriSpeech
tsv_dir=data/tsv

python prepare_data/main.py \
    -d librispeech \
    -r $libri_root \
    -o $tsv_dir