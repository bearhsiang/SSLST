#!/usr/local/bin bash

dataset=libritrans
data_root=/hdd/libritrans
src_lang=en
tgt_lang=fr
output_root=data/tsv/

python prepare_data/main.py \
    -d $dataset \
    -r $data_root \
    -s $src_lang \
    -t $tgt_lang \
    -o $output_root