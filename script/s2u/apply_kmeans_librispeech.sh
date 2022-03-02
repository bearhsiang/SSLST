#!/usr/local/bin bash 

dataset=librispeech
splits="train-clean-100"
audio_key=audio
librispeech_root=/hdd/LibriSpeech
model_path=data/ssl_models/hubert_base_ls960.pt
model_name=hubert
layer=9
km_model_path=data/kmeans_model/hubert_base_ls960_L9_km500.bin
suffix=hubert_l9_km500

for split in $splits; do
    bash script/s2u/apply_kmeans.sh \
        --dataset $dataset \
        --split $split \
        --audio-key $audio_key \
        --audio-root $librispeech_root \
        --model-path $model_path \
        --model-name $model_name \
        --layer $layer \
        --km-model-path $km_model_path \
        --suffix $suffix
done