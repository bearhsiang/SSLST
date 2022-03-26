#!/usr/local/bin bash 

source script/setup.sh

dataset=libritrans-en-fr
splits="train"
audio_key=audio

for split in $splits; do
    python utils_new/create_fs_manifest.py \
        --audio-root $sslst_libritrans_root \
        --tsv-file $sslst_data_root/tsv/$dataset/$split.tsv \
        --key $audio_key \
        --output $sslst_data_root/$dataset/manifest/$split.tsv
done