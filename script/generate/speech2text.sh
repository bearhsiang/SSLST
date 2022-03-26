#!/usr/local/bin bash

source script/setup.sh

name=$1
src_lang=$2
tgt_lang=$3

arch=s2t_transformer_s

fairseq-generate \
    $sslst_data_root/$name/s2t-$src_lang-$tgt_lang \
    --config-yaml config.yaml \
    --gen-subset test --task speech_to_text \
    --path $sslst_output_root/$name/$arch-$src_lang-$tgt_lang/checkpoint_best.pt \
    --beam 5 \
    --max-len-a 0 \
    --max-len-b 256 \
    --scoring sacrebleu \
    --max-source-positions 4096 \
    # --skip-invalid-size-inputs-valid-test
