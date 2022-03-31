#!/usr/local/bin bash

source script/setup.sh

dataset=$1
model=$2
layer=$3
format=$4 # collect | seperate

nshard=5

splits="train dev test"
device=cuda

for split in $splits; do 
    for rank in `seq 0 $((nshard -1))`; do
        if [ ! -f $sslst_feat_root/$dataset/$model/$layer/${split}_${rank}_${nshard}.npy ] || [ $format != "collect" ]; then
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
        else
            echo $sslst_feat_root/$dataset/$model/$layer/${split}_${rank}_${nshard}.npy exists, skip...
        fi
    done
done