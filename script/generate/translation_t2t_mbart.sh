#!/usr/local/bin bash

source script/setup.sh

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

dataset=$1
src_lang=$2
tgt_lang=$3
arch=mbart_large

fairseq-generate \
    $sslst_data_bin_root/$dataset/mbart-$src_lang-$tgt_lang \
    --user-dir fairseq-src \
    --task sslst_translation_from_pretrained_bart \
    --langs $langs \
    --gen-subset test \
    --path $sslst_output_root/$dataset/mbart-$arch-$src_lang-$tgt_lang/checkpoint_best.pt \
    --beam 5 \
    --max-len-a 1.5 \
    --max-len-b 20 \
    --scoring sacrebleu \
    --max-source-positions 1024 \
    --remove-bpe sentencepiece \
    --fp16 \
    # --skip-invalid-size-inputs-valid-test
