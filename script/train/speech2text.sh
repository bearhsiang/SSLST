
source script/setup.sh

name=$1
src_lang=$2
tgt_lang=$3

arch=s2t_transformer_s

python -m fairseq_cli.train \
    $sslst_data_root/$name/s2t-$src_lang-$tgt_lang \
    --user-dir fairseq-src \
    --task speech_to_text \
    --config-yaml config.yaml \
    --train-subset train \
    --valid-subset dev \
    --save-dir $sslst_output_root/$name/$arch-$src_lang-$tgt_lang \
    --num-workers 4 \
    --max-update 32000 \
    --max-tokens 20000 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --report-accuracy \
    --arch $arch --optimizer adam --lr 2e-3 \
    --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed $sslst_seed --update-freq 16 \
    --wandb-project sslst-$name \
    --keep-last-epochs 3 \
    --fp16
