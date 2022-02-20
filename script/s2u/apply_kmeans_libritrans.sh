#!/usr/local/bin bash 

dataset=libritrans-en-fr
splits="test"
audio_key=audio
libritrans_root=/hdd/libritrans
ssl_model_path=data/ssl_models/hubert_base_ls960.pt
layer=9
km_model_path=data/kmeans_model/hubert_base_ls960_L9_km500.bin
suffix=hubert_l9_km500

for split in $splits; do
    bash script/s2u/apply_kmeans.sh \
        --dataset $dataset \
        --split $split \
        --audio-key $audio_key \
        --audio-root $libritrans_root/$split/audiofiles \
        --ssl-model-path $ssl_model_path \
        --layer $layer \
        --km-model-path $km_model_path \
        --suffix $suffix
done