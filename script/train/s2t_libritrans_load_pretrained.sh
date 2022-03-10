
source script/setup.sh

name=libritrans-en-fr

arch=s2t_transformer_bart_base
src_lang=hubert_l9
tgt_lang=fr

# pt_model_path=$sslst_data_root/mbart.cc25.v2/model.pt
pt_model_path=$sslst_data_root/bart.base/model.pt
pt_model_arch=bart_base

fairseq-train \
    $sslst_data_root/$name/s2t-$src_lang-$tgt_lang \
    --user-dir fairseq-src \
    --task speech_to_text \
    --config-yaml config.yaml \
    --train-subset train \
    --valid-subset dev \
    --save-dir $sslst_output_root/$name/ft-$arch-$src_lang-$tgt_lang \
    --num-workers 4 \
    --max-update 32000 \
    --max-tokens 20000 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --report-accuracy \
    --arch $arch --optimizer adam --lr 2e-3 \
    --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
    --pt-encoder-path $pt_model_path \
    --pt-encoder-arch $pt_model_arch \
    --pt-decoder-path $pt_model_path \
    --pt-decoder-arch $pt_model_arch \
    --wandb-project sslst-$name

    # --fp16 \

    # --load-pretrained-encoder-from $pt_model_path \
    
    # --encoder-freezing-updates 1000 
    # --load-pretrained-encoder-from ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}
