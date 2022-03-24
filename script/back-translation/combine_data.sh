#!/usr/local/bin bash

source script/setup.sh

dataset=libritrans-en-fr
splits="train dev test"
src_lang=hubert_l9_km500_simple
tgt_lang=fr
bt_name=newscrawl_fr
bt_subset=max30-min10-N10000

new_dataset=$dataset-$bt_name-$bt_subset

mkdir $sslst_data_root/$new_dataset

for split in $splits; do
    for lang in $src_lang $tgt_lang; do
        cp $sslst_data_root/$dataset/$split.$lang $sslst_data_root/$new_dataset
    done
done

sed 's/^/[BT] /' $sslst_data_root/$dataset/$bt_name/$bt_subset.gen.$src_lang >> $sslst_data_root/$new_dataset/train.$src_lang

cat $sslst_data_root/$dataset/$bt_name/$bt_subset.$tgt_lang >> $sslst_data_root/$new_dataset/train.$tgt_lang