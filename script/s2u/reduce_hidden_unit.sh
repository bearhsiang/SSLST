#!/usr/local/bin bash

source script/setup.sh

dataset=libritrans-en-fr
suffix=hubert_l9_km100
splits="train dev test"
mode=simple

for split in $splits; do
    python utils_new/reduce_hidden_unit.py \
        -i $sslst_data_root/$dataset/$split.$suffix \
        -o $sslst_data_root/$dataset/$split.${suffix}_${mode} \
        -m $mode
    echo finish reducing $sslst_data_root/$dataset/$split.$suffix, first example:
    echo "<<"
    head $sslst_data_root/$dataset/$split.$suffix -n 1
    echo ">>"
    head $sslst_data_root/$dataset/$split.${suffix}_${mode} -n 1
    echo "---"
done