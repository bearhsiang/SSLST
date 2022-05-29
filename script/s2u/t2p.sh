#!/usr/local/bin bash

source script/setup.sh

dataset=$1
lang=$2

splits="train dev test"
data_dir=$sslst_data_root/$dataset

for split in $splits; do
    python utils_new/g2p.py \
        --input $data_dir/tmp/$split.filtered.$lang \
        --output $data_dir/$split.phoneme
done