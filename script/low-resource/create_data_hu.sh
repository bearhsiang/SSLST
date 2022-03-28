#!/usr/local/bin bash

source script/setup.sh

dataset=$1
hr=$2
lang=$3

ori_dir=$sslst_data_root/$dataset
tgt_dir=$sslst_data_root/$dataset-hr$hr

# hu
python utils_new/select_with_ids.py \
    --selected-ids-file $tgt_dir/selected_ids.txt \
    --input $ori_dir/train.$lang \
    --output $tgt_dir/train.$lang

for split in dev test; do
    cp $ori_dir/$split.$lang $tgt_dir/$split.$lang
done