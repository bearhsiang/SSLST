#!/usr/local/bin bash

source script/setup.sh

dataset=covost2
src_lang=de
tgt_lang=en

python prepare_data/main.py \
    -d $dataset \
    -r $sslst_covost2_tsv_root \
    -s $src_lang \
    -t $tgt_lang \
    -o $sslst_data_root/tsv