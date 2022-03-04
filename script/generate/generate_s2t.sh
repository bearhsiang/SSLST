#!/usr/local/bin bash

data_root=data
name=libritrans-en-fr

arch=s2t_transformer_s
src_lang=hubert_l9
tgt_lang=fr

fairseq-generate \
    $data_root/$name/s2t-$src_lang-$tgt_lang \
    --config-yaml config.yaml \
    --gen-subset test --task speech_to_text \
    --path $sslst_output_root/$name/$arch-$src_lang-$tgt_lang/checkpoint_best.pt \
    --beam 5 \
    --max-len-a 0 \
    --max-len-b 256 \
    --max-source-positions 4096 \
    --scoring sacrebleu

    # --skip-invalid-size-inputs-valid-test
