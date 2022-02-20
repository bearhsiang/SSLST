#!/usr/local/bin bash

name=libritrans-en-fr
data_root=data
feat_root=/hdd/ssl_feat
model_name=hubert
layer=9
nshard=5

for split in train; do
    python utils_new/split_feature.py \
        --feat-dir $feat_root/$name/$model_name/$layer \
        --split $split \
        --nshard $nshard \
        --manifest $data_root/$name/manifest/$split.tsv \
        --out-feat-dir $feat_root/$name/$model_name/$layer/$split
done