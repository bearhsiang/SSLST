#!/usr/local/bin bash

source script/setup.sh

name=libritrans-en-fr
src_lang=hubert_l9_km500
tgt_lang=fr

data_dir=$sslst_data_root/$name
data_bin_dir=$sslst_data_bin_root/$name/$src_lang-$tgt_lang

fairseq-preprocess \
    -s $src_lang \
    -t $tgt_lang \
    --trainpref $data_dir/train \
    --validpref $data_dir/dev \
    --testpref $data_dir/test \
    --destdir $data_bin_dir