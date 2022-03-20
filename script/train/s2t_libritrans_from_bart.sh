
source script/setup.sh

name=libritrans-en-fr

arch=s2t_transformer_mbart_large
src_lang=hubert_l9
tgt_lang=fr

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

pt_model_path=$sslst_data_root/mbart.cc25.v2/model.pt
pt_model_arch=mbart_large

fairseq-train \
    $sslst_data_root/$name/s2t-mbart-$src_lang-$tgt_lang \
    --user-dir fairseq-src \
    --task speech_to_text_from_pretrained_bart \
    --langs $langs \
    --config-yaml config.yaml \
    --prepend-bos \
    --train-subset train \
    --valid-subset dev \
    --save-dir $sslst_output_root/$name/ft-test-$pt_model_arch-$arch-$src_lang-$tgt_lang \
    --num-workers 4 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 2500 --total-num-update 40000 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --max-tokens 8192 --update-freq 2 \
    --report-accuracy \
    --arch $arch \
    --seed $sslst_seed \
    --keep-last-epochs 3 \
    --pt-encoder-path $pt_model_path \
    --pt-encoder-arch $pt_model_arch \
    --pt-decoder-path $pt_model_path \
    --pt-decoder-arch $pt_model_arch \
    --ignore-prefix-size 1 \
    # --wandb-project sslst-$name \
    # --fp16
