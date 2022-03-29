#!/usr/local/bin bash

source script/setup.sh

name=$1
model_name=$2
layer=$3
nshard=5

for split in train dev test; do
    python utils_new/split_feature.py \
        --feat-dir $sslst_feat_root/$name/$model_name/$layer \
        --split $split \
        --nshard $nshard \
        --manifest $sslst_data_root/$name/manifest/$split.tsv \
        --out-feat-dir $sslst_feat_root/$name/$model_name/$layer/$split
done