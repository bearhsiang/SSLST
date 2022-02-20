#!/usr/local/bin bash

data_root=data
dataset=libritrans-en-fr
suffix=hubert_l9_km500
splits="train dev test"
mode=addN

for split in $splits; do
    python utils_new/reduce_hidden_unit.py \
        -i $data_root/$dataset/$split.$suffix \
        -o $data_root/$dataset/$split.${suffix}_${mode} \
        -m $mode
    echo finish reducing $data_root/$dataset/$split.$suffix, first example:
    echo "<<"
    head $data_root/$dataset/$split.$suffix -n 1
    echo ">>"
    head $data_root/$dataset/$split.${suffix}_${mode} -n 1
    echo "---"
done