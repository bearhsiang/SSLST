#!/usr/local/bin bash

source script/setup.sh

mbart_root=$sslst_data_root/mbart.cc25.v2

dataset=libritrans-en-fr-hr100
splits="train dev test"
src_lang=en
mbart_src_lang=en_XX
tgt_lang=fr
mbart_tgt_lang=fr_XX

for split in $splits; do
    spm_encode \
        --model $mbart_root/sentence.bpe.model \
        < $sslst_data_root/$dataset/tmp/$split.filtered.$src_lang \
        > $sslst_data_root/$dataset/$split.$mbart_src_lang

    spm_encode \
        --model $mbart_root/sentence.bpe.model \
        < $sslst_data_root/$dataset/tmp/$split.filtered.$tgt_lang \
        > $sslst_data_root/$dataset/$split.$mbart_tgt_lang
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