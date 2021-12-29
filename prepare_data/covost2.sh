#!/usr/local/bin bash

dataset=covost2
data_root=/hdd/covost/tsv/
src_lang=en
tgt_lang=de
output_root=data/tsv/

python prepare_data/main.py \
    -d $dataset \
    -r $data_root \
    -s $src_lang \
    -t $tgt_lang \
    -o $output_root