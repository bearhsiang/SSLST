#!/usr/local/bin bash

export sslst_seed=1
export sslst_sample_rate=16000

export sslst_data_root=/hdd/sslst_data
export sslst_feat_root=/ssd/sslst_feat
export sslst_output_root=/hdd/sslst_results
export sslst_data_bin_root=/hdd/sslst_databin

# dataset 
export sslst_libritrans_root=/hdd/libritrans
export sslst_librispeech_root=/hdd/LibriSpeech
export sslst_cv_version=cv-corpus-8.0-2022-01-19
export sslst_covost2_tsv_root=/hdd/covost/tsv
export sslst_covost2_root=/hdd/covost/$sslst_cv_version

# repo
export sslst_mosesdecoder_root=~/Desktop/SSLST/mosesdecoder
export sslst_fairseq_root=~/Desktop/SSLST/fairseq
export sslst_covost_root=~/Desktop/SSLST/covost