#!/usr/local/bin bash

source script/setup.sh

dataset=libritrans-en-fr
wandb_name=sslst-libritrans-en-fr
src_lang=$1
tgt_lang=$2

arch=transformer_iwslt_de_en

data_dir=$sslst_data_bin_root/$dataset/$src_lang-$tgt_lang

fairseq-train \
    $data_dir \
    --source-lang $src_lang --target-lang $tgt_lang \
    --arch $arch \
    --max-update 100000 \
    --save-dir $sslst_output_root/$dataset/$arch-$src_lang-$tgt_lang-f8 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 5120 \
    --max-source-positions 4096 \
    --max-target-positions 512 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 0, "max_len_b": 256}' \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --wandb-project $wandb_name \
    --keep-last-epochs 3 \
    --update-freq 8 \
    --fp16

    # --share-decoder-input-output-embed \
    # --validate-interval-updates 5000 \
    # --patience 20
    # --patience 10 
    # --max-source-positions 4096 \
    # --skip-invalid-size-inputs-valid-test \
    # --eval-bleu-detok moses \


