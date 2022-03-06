#!/usr/local/bin bash

source script/setup.sh
mbart_root=$sslst_data_root/mbart.cc25.v2
mode=random

dataset=libritrans-en-fr
split=train
lang=hubert_l9_km500_simple

python utils_new/create_hidden_unit_mapping.py \
    --input $sslst_data_root/$dataset/$split.$lang \
    --dict $mbart_root/dict.txt \
    --output $sslst_data_root/$dataset/$lang-mbart.$mode \
    --mode $mode