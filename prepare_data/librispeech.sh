#!/usr/local/bin bash

source script/setup.sh

dataset=librispeech
data_root=$sslst_librispeech_root
output_root=$sslst_data_root/tsv

python prepare_data/main.py \
    -d $dataset \
    -r $data_root \
    -o $output_root