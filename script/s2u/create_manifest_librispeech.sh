#!/usr/local/bin bash 

source script/setup.sh

dataset=librispeech
audio_key=audio
splits="train-clean-100 dev-clean test-clean"

for split in $splits; do
    python utils_new/create_fs_manifest.py \
        --audio-root $sslst_librispeech_root \
        --tsv-file $sslst_data_root/tsv/$dataset/$split.tsv \
        --key $audio_key \
        --output $sslst_data_root/$dataset/manifest/$split.tsv \
        --sample-rate $sslst_sample_rate
done