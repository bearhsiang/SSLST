#!/usr/local/bin bash

source script/setup.sh

name=$1
src_lang=$2
tgt_lang=$3

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
pretrained_model=$sslst_data_root/mbart.cc25.v2/model.pt

data_dir=$sslst_data_bin_root/$name/mbart-$src_lang-$tgt_lang
arch=mbart_large

fairseq-train \
    $data_dir \
    --user-dir fairseq-src \
    --encoder-normalize-before --decoder-normalize-before \
    --arch $arch --layernorm-embedding \
    --task sslst_translation_from_pretrained_bart \
    --source-lang $src_lang --target-lang $tgt_lang \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 2500 --total-num-update 40000 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --max-tokens 1024 --update-freq 2 \
    --keep-last-epochs 3 \
    --seed 222 --log-format simple --log-interval 2 \
    --restore-file $pretrained_model \
    --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
    --langs $langs \
    --ddp-backend legacy_ddp \
    --fp16 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.5, "max_len_b": 20}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --wandb-project sslst-$name \
    --save-dir $sslst_output_root/$name/mbart-$arch-$src_lang-$tgt_lang
