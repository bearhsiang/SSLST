#!/usr/local/bin bash

python convert/s2u_train_kmeans.py \
    -u hubert \
    -i data/s2t/libritrans-en-fr/normalized/test.tsv \
    -k audio \
    -d /hdd/libritrans/test/audiofiles \
    -l 6 \
    -o tmp \
    -c \
    -f 0.01
