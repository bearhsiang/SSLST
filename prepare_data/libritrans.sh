#!/usr/local/bin bash

source script/setup.sh

dataset=libritrans
data_root=$sslst_libritrans_root
src_lang=en
tgt_lang=fr
output_root=$sslst_data_root/tsv

python prepare_data/main.py \
    -d $dataset \
    -r $data_root \
    -s $src_lang \
    -t $tgt_lang \
    -o $output_root