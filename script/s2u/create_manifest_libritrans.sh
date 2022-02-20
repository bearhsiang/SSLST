#!/usr/local/bin bash 

dataset=libritrans-en-fr
data_root=data
tsv_root=data/tsv
splits="train dev test"
audio_key=audio
libritrans_root=/hdd/libritrans

for split in $splits; do
    python utils_new/create_fs_manifest.py \
        --audio-root $libritrans_root \
        --tsv-file $tsv_root/$dataset/$split.tsv \
        --key $audio_key \
        --output $data_root/$dataset/manifest/$split.tsv
done