#!/usr/local/bin bash

source script/setup.sh

dataset=$1
src_lang=$2
tgt_lang=$3

arch=transformer_iwslt_de_en

fairseq-generate \
    $sslst_data_bin_root/$dataset/$src_lang-$tgt_lang \
    --gen-subset test \
    --path $sslst_output_root/$dataset/$arch-$src_lang-$tgt_lang/checkpoint_best.pt \
    --beam 5 \
    --max-len-a 0 \
    --max-len-b 256 \
    --scoring sacrebleu \
    --max-source-positions 4096 \
    --remove-bpe sentencepiece \
    # --skip-invalid-size-inputs-valid-test

