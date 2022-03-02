#!/usr/local/bin bash

source script/setup.sh

dataset=libritrans-en-fr
format=collect

# model=modified_cpc
model=$1
# layer=1
layer=$2
nshard=5

splits='train dev test'
device='cuda'

for split in $splits; do 
    for rank in `seq 0 $((nshard -1))`; do
        python utils_new/dump_features.py \
            --manifest $sslst_data_root/$dataset/manifest \
            --split $split \
            --model $model \
            --layer $layer \
            --nshard $nshard \
            --rank $rank \
            --device $device \
            --output-dir $sslst_feat_root/$dataset/$model/$layer \
            --format $format
    done
done