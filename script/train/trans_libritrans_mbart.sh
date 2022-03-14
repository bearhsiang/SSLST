#!/usr/local/bin bash

source script/setup.sh

name=$1
src_lang=$2
tgt_lang=$3

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
pretrained_model=$sslst_data_root/mbart.cc25.v2/model.pt

data_dir=$sslst_data_bin_root/$name/mbart-$src_lang-$tgt_lang

fairseq-train \
    $data_dir \
    --user-dir fairseq-src \
    --encoder-normalize-before --decoder-normalize-before \
    --arch mbart_large --layernorm-embedding \
    --task sslst_translation_from_pretrained_bart \
    --source-lang $src_lang --target-lang $tgt_lang \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 2500 --total-num-update 40000 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --max-tokens 1024 --update-freq 2 \
    --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
    --seed $sslst_seed --log-format simple --log-interval 2 \
    --restore-file $pretrained_model \
    --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
    --langs $langs \
    --ddp-backend legacy_ddp \
    --keep-last-epochs 3 \
    --fp16


# fairseq-train \
#     $data_dir \
#     --source-lang $src_lang --target-lang $tgt_lang \
#     --arch $arch \
#     --save-dir $output_root/$name/$arch-$src_lang-$tgt_lang \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#     --dropout 0.3 --weight-decay 0.0001 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 5120 \
#     --max-source-positions 4096 \
#     --fp16 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 20}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe sentencepiece \
#     --eval-bleu-print-samples \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#     --skip-invalid-size-inputs-valid-test \
#     --wandb-project sslst-$name \
#     --keep-last-epochs 3 \
#     --share-decoder-input-output-embed \
#     --validate-interval-updates 5000


    # --patience 20
    # --patience 10 

