#!/usr/local/bin bash

data_bin_root=data-bin
output_root=/hdd/sslst_results
ft_path=$output_root/libritrans-en-fr/transformer_iwslt_de_en-en-fr/checkpoint_best.pt

name=libritrans-en-fr
src_lang=hubert_l9_km500
tgt_lang=fr

arch=transformer_iwslt_de_en

data_dir=$data_bin_root/$name/$src_lang-$tgt_lang-ft

fairseq-train \
    $data_dir \
    --finetune-from-model $ft_path \
    --source-lang $src_lang --target-lang $tgt_lang \
    --arch $arch \
    --save-dir $output_root/$name/$arch-$src_lang-$tgt_lang-f8-ft \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 5120 \
    --max-source-positions 4096 \
    --max-target-positions 4096 \
    --fp16 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 0, "max_len_b": 4096}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --wandb-project sslst-$name \
    --keep-last-epochs 3 \
    --update-freq 8

    # --share-decoder-input-output-embed \
    # --validate-interval-updates 5000 \
    # --patience 20
    # --patience 10 
    # --max-source-positions 4096 \

