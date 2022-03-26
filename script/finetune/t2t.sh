#!/usr/local/bin bash

source script/setup.sh

dataset=libritrans-en-fr
splits="train dev test"
pretrain_src_lang=en
src_lang=hubert_l9_km500_simple
tgt_lang=fr
mode=random

dict=$sslst_data_bin_root/$dataset/$pretrain_src_lang-$tgt_lang/dict.$pretrain_src_lang.txt

new_src_lang=${src_lang}_${pretrain_src_lang}_$mode

python utils_new/create_hidden_unit_mapping.py \
    --input $sslst_data_root/$dataset/train.$src_lang \
    --dict $dict \
    --mode $mode \
    --output $sslst_data_root/$dataset/$src_lang-$pretrain_src_lang.$mode

for split in $splits; do
    python utils_new/map_hidden_unit.py \
        --input $sslst_data_root/$dataset/$split.$src_lang \
        --mapping $sslst_data_root/$dataset/$src_lang-$pretrain_src_lang.$mode \
        --output $sslst_data_root/$dataset/$split.$new_src_lang
done

fairseq-preprocess \
    -s $new_src_lang \
    -t $tgt_lang \
    --trainpref $sslst_data_root/$dataset/train \
    --validpref $sslst_data_root/$dataset/dev \
    --testpref $sslst_data_root/$dataset/test \
    --destdir $sslst_data_bin_root/$dataset/ft-${new_src_lang}-${tgt_lang} \
    --srcdict $sslst_data_bin_root/$dataset/$pretrain_src_lang-$tgt_lang/dict.$pretrain_src_lang.txt \
    --tgtdict $sslst_data_bin_root/$dataset/$pretrain_src_lang-$tgt_lang/dict.$tgt_lang.txt