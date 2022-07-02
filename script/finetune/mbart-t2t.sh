#!/usr/local/bin bash

source script/setup.sh

mbart_root=$sslst_data_root/mbart.cc25.v2

dataset=libritrans-en-fr
splits="train dev test"
src_lang=en
mbart_src_lang=en_XX
tgt_lang=fr
mbart_tgt_lang=fr_XX

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

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

python -m fairseq_cli.preprocess \
    -s $mbart_src_lang \
    -t $mbart_tgt_lang \
    --trainpref $sslst_data_root/$dataset/train \
    --validpref $sslst_data_root/$dataset/dev \
    --testpref $sslst_data_root/$dataset/test \
    --destdir $sslst_data_bin_root/$dataset/mbart-${mbart_src_lang}-${mbart_tgt_lang} \
    --srcdict $mbart_root/dict.txt \
    --tgtdict $mbart_root/dict.txt