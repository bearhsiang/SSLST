
source script/setup.sh

name=libritrans-en-fr

arch=s2t_transformer_mbart_large
src_lang=hubert_l9
tgt_lang=fr

pt_model_path=$sslst_data_root/mbart.cc25.v2/model.pt

fairseq-train \
    $sslst_data_root/$name/s2t-$src_lang-$tgt_lang \
    --user-dir fairseq-src \
    --task speech_to_text \
    --config-yaml config.yaml \
    --train-subset train \
    --valid-subset dev \
    --save-dir $sslst_output_root/$name/ft-test-$arch-$src_lang-$tgt_lang \
    --num-workers 4 \
    --max-update 100000 \
    --max-tokens 20000 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --report-accuracy \
    --arch $arch --optimizer adam --lr 2e-3 \
    --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 16 \
    --fp16 \
    --pt-encoder-path $pt_model_path \
    --pt-encoder-arch mbart_large \
    --pt-decoder-path $pt_model_path \
    --pt-decoder-arch mbart_large
    # --load-pretrained-encoder-from $pt_model_path \
    # --wandb-project sslst-$name

    
    # --encoder-freezing-updates 1000 
    # --load-pretrained-encoder-from ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}
