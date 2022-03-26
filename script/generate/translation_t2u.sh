#!/usr/local/bin bash 

source script/setup.sh

dataset=$1
src_lang=$2
tgt_lang=$3
arch=transformer_iwslt_de_en

fairseq-generate \
    $sslst_data_bin_root/$dataset/$tgt_lang-$src_lang \
    -s $src_lang -t $tgt_lang \
    --gen-subset test \
    --path $sslst_output_root/$dataset/$arch-$src_lang-$tgt_lang-f8/checkpoint_best.pt \
    --beam 1 \
    --max-len-a 8 \
    --max-len-b 20 \
    --scoring sacrebleu \
    --max-target-positions 4096 \
    # --skip-invalid-size-inputs-valid-test \
