#!/usr/local/bin bash

data_bin_root=data-bin
ckpt_root=/hdd/sslst_results/
name=libritrans-en-fr
src_lang=en
tgt_lang=fr
arch=transformer_iwslt_de_en

beam=5

fairseq-generate \
    $data_bin_root/$name/$src_lang-$tgt_lang \
    -s $src_lang -t $tgt_lang \
    --gen-subset test \
    --path $ckpt_root/$name/$src_lang-$tgt_lang/$arch/checkpoint_best.pt \
    --max-source-positions 1000 \
    --max-tokens 4096 --beam $beam --scoring sacrebleu \
    --remove-bpe sentencepiece