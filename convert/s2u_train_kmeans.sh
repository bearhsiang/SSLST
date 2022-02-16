#!/usr/local/bin bash

split=train
layer=0
model=fbank
n_clusters=50
bsz=5
frac=0.01
tsv=data/s2t/libritrans-en-fr/normalized/$split.tsv
audio_dir=/hdd/libritrans/$split/audiofiles


python convert/s2u_train_kmeans.py \
    -u $model \
    -i $tsv \
    -k audio \
    -d $audio_dir \
    -l $layer \
    -o data/kmeans_model/libritrans_en_fr-$split-$frac-$model-$layer-$n_clusters \
    -c \
    -f $frac \
    -b $bsz \
    -n $n_clusters
