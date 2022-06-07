#!/usr/local/bin bash

source script/setup.sh

dataset=$1
src_lang=$2
tgt_lang=$3
bpe=8000

arch=transformer_iwslt_de_en
ckpt=$sslst_output_root/$dataset/$arch-$src_lang-$tgt_lang/checkpoint_best.pt

out_dir=$sslst_data_bin_root/$dataset/$src_lang-$tgt_lang

fairseq-generate \
    $sslst_data_bin_root/$dataset/$src_lang-$tgt_lang \
    --gen-subset test \
    --path $ckpt \
    --beam 5 \
    --max-len-a 0 \
    --max-len-b 256 \
    --max-source-positions 4096 \
    --remove-bpe sentencepiece \
    --max-tokens 8192 \
    --scoring sacrebleu 2>&1 | tee $out_dir/test.$src_lang-$tgt_lang.out

cat $out_dir/test.$src_lang-$tgt_lang.out | grep ^D | LC_ALL=C sort -V | cut -f3- > $out_dir/test.$src_lang-$tgt_lang.out.$tgt_lang