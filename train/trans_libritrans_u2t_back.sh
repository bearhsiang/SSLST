#!/usr/local/bin bash

data_bin_root=data-bin
output_root=/hdd/sslst_results

name=libritrans-en-fr
src_lang=fr
tgt_lang=hubert_l9_km500_simple

arch=transformer_iwslt_de_en

data_dir=$data_bin_root/$name/$tgt_lang-$src_lang

fairseq-train \
    $data_dir \
    --source-lang $src_lang --target-lang $tgt_lang \
    --arch $arch \
    --save-dir $output_root/$name/$arch-$src_lang-$tgt_lang-f8 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-source-positions 1024 \
    --max-target-positions 4096 \
    --fp16 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 1, "max_len_a": 0, "max_len_b": 512}' \
    --eval-bleu-detok moses \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --wandb-project sslst-$name \
    --keep-last-epochs 3 \
    --update-freq 8

    # --share-decoder-input-output-embed \
    # --patience 20
    # --patience 10 
    # --max-source-positions 4096 \
    # --eval-bleu-remove-bpe sentencepiece \

