#!/usr/local/bin bash

data_root=data/trans
data_bin_root=data-bin

name=libritrans-en-fr
src_lang=logmel_50_simple
tgt_lang=fr

data_dir=$data_root/$name/$src_lang-$tgt_lang
data_bin_dir=$data_bin_root/$name/$src_lang-$tgt_lang

fairseq-preprocess \
    -s $src_lang \
    -t $tgt_lang \
    --trainpref $data_dir/train \
    --validpref $data_dir/dev \
    --testpref $data_dir/test \
    --destdir $data_bin_dir