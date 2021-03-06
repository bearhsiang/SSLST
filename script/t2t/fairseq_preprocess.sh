#!/usr/local/bin bash

source script/setup.sh

name=$1
src_lang=$2
tgt_lang=$3

data_dir=$sslst_data_root/$name
data_bin_dir=$sslst_data_bin_root/$name/$src_lang-$tgt_lang

mkdir -p $data_bin_dir

python -m fairseq_cli.preprocess \
    -s $src_lang \
    -t $tgt_lang \
    --trainpref $data_dir/train \
    --validpref $data_dir/dev \
    --testpref $data_dir/test \
    --destdir $data_bin_dir \
    --workers 4 \
    | tee $data_bin_dir/preprocess.log