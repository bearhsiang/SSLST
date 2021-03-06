#!/usr/local/bin bash

source script/setup.sh

dataset=librispeech
format=collect

model=$1
layer=$2

nshard=5

splits='train-clean-100'
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