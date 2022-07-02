#!/usr/local/bin bash

source script/setup.sh

name=$1
src_lang=$2
tgt_lang=$3

arch=transformer_iwslt_de_en
data_dir=$sslst_data_bin_root/$name/$src_lang-$tgt_lang
save_dir=$sslst_output_root/$name/$arch-$src_lang-$tgt_lang
wandb_name=sslst-$name

python -m fairseq_cli.train \
    $data_dir \
    --source-lang $src_lang --target-lang $tgt_lang \
    --arch $arch \
    --save-dir $save_dir \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-update 40000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 5120 \
    --max-source-positions 4096 \
    --fp16 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.5, "max_len_b": 20}' \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --wandb-project $wandb_name \
    --keep-last-epochs 3
