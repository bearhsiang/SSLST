#!/usr/local/bin bash

source script/setup.sh

mbart_root=$sslst_data_root/mbart.cc25.v2

dataset=libritrans-en-fr
splits='train dev test'
src_lang=hubert_l9_km500_simple
mode=random
mbart_src_lang=${src_lang}_mbart_${mode}
mbart_tgt_lang=fr_XX

mapping_file=$sslst_data_root/$dataset/$src_lang-mbart.$mode

for split in $splits; do
    python utils_new/map_hidden_unit.py \
        --input $sslst_data_root/$dataset/$split.$src_lang \
        --mapping $mapping_file \
        --output $sslst_data_root/$dataset/$split.$mbart_src_lang
done

fairseq-preprocess \
    -s $mbart_src_lang \
    -t $mbart_tgt_lang \
    --trainpref $sslst_data_root/$dataset/train \
    --validpref $sslst_data_root/$dataset/dev \
    --testpref $sslst_data_root/$dataset/test \
    --destdir $sslst_data_bin_root/$dataset/mbart-${mbart_src_lang}-${mbart_tgt_lang} \
    --srcdict $mbart_root/dict.txt \
    --tgtdict $mbart_root/dict.txt