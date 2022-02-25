#!/usr/local/bin bash

dataset=librispeech
data_root=data
feat_root=/ssd/ssl_feat

# model=modified_cpc
model=$1
# layer=1
layer=$2
nshard=5

splits='train-clean-100'
device='cuda'

for split in $splits; do 
    for rank in `seq 0 $((nshard -1))`; do
        python utils_new/dump_features.py \
            --manifest $data_root/$dataset/manifest \
            --split $split \
            --model $model \
            --layer $layer \
            --nshard $nshard \
            --rank $rank \
            --device $device \
            --output-dir $feat_root/$dataset/$model/$layer
    done
done