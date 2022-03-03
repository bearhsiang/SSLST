
source script/setup.sh

name=libritrans-en-fr

arch=s2t_transformer_s
src_lang=hubert_l6
tgt_lang=fr

fairseq-train \
    $sslst_data_root/$name/s2t-$src_lang-$tgt_lang \
    --task speech_to_text \
    --config-yaml config.yaml \
    --train-subset train \
    --valid-subset dev \
    --save-dir $sslst_output_root/$name/$arch-$src_lang-$tgt_lang \
    --num-workers 4 \
    --max-update 100000 \
    --max-tokens 20000 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --report-accuracy \
    --arch $arch --optimizer adam --lr 2e-3 \
    --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 16 \
    --fp16 \
    --wandb-project sslst-$name

    
    # --encoder-freezing-updates 1000 
    # --load-pretrained-encoder-from ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}
