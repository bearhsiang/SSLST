#!/usr/local/bin bash

source script/setup.sh

dataset=libritrans-en-fr
src_lang=fbank_l0_km500_simple
tgt_lang=fr
arch=transformer_iwslt_de_en

fairseq-generate \
    $sslst_data_bin_root/$dataset/$src_lang-$tgt_lang \
    --gen-subset test \
    --path $sslst_output_root/$dataset/$arch-$src_lang-$tgt_lang-f8/checkpoint_best.pt \
    --beam 5 \
    --max-len-a 0 \
    --max-len-b 256 \
    --scoring sacrebleu \
    --max-source-positions 4096 \
    --remove-bpe sentencepiece

    # --skip-invalid-size-inputs-valid-test \
