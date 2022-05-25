# SSLST

## Setup

Please first setup the following configuration in `script/setup.sh`. (You could do this by copying and modifying `script/setup_example.sh`)
* `sslst_data_root`: The place to put the processed data.
* `sslst_feat_root`: The place to store the extracted features. (A large storage would be needed.)
* `sslst_output_root`: The place to store the checkpoints.
* `sslst_data_bin_root`: The place to store binarized text data.

## Get data

The speech dataset will be processed into tsv files. This repo default supports three common dataset, e.g. librispeech, libritrans and covost2, and also allow you to add new dataset.

### To prepare supported dataset

* Librispeech
1. Download Librispeech from [Official website](https://www.openslr.org/12).
We set `train-clean-100`, `dev-clean`, `test-clean` as default splits, modifying `prepare_data/Librispeech.py` to change this setting.
2. Set `$sslst_librispeech_root` in `script/setup.sh`
3. Run `bash prepare_data/librispeech.sh`


* Libritrans
1. Download Libritrans from [this repo](https://github.com/alicank/Translation-Augmented-LibriSpeech-Corpus)
1. Set `$sslst_libritrans_root` in `script/setup.sh`
2. Run `bash prepare_data/libritrans.sh`

* CoVoST2
1. Download Common Voice from [Official Website](https://commonvoice.mozilla.org/en/datasets) and set `$sslst_cv_version` in `script/setup.sh`. You should choose the source language based on the translation direction. **Note that we use De -> En as default translation direction.**
2. Clone [CoVoST](https://github.com/facebookresearch/covost) and set `$sslst_covost_root` and `$sslst_covost2_tsv_root` in `script/setup.sh`
3. Run `bash prepare_data/pre_covost2.sh` and `bash prepare_data/covost2.sh`. 

**The data would be prepared at `$sslst_data_root/tsv` in tsv format.**

### To add new dataset

Use `prepare_data/Example.py` as template to add new dataset, and it would be automatically detected.

## Preprocessing

### Text

For all the text data, we do the following preprocessing steps.
1. Normalize punctuations
2. Remove unprintable characters
3. Lowercase all the characters
4. Build BPE tokenizer with size = 8000 and character coverage = 1

To do the preprocessing
1. Clone [mosesdecoder](https://github.com/moses-smt/mosesdecoder) and set `$sslst_mosesdecoder_root` in `script/setup.sh`.
2. Run `bash script/t2t/[DATASET].sh` to do the preprocessing for the dataset you want to use.

### Speech

#### Speech to hidden unit

1. Create manifest by `bash script/s2u/create_manifest_[DATASET].sh`
2. Clone and install [Fairseq](https://github.com/facebookresearch/fairseq). Set `$sslst_fairseq_root` in `script/setup.sh`
3. Train K-means model by `bash script/s2u/train_kmeans_simple.sh [DATASET] [KM_TAG] [SSL_MODEL] [LAYER] [N_CLUSTER] [PERCENTAGE]`. The kmeans model could be found as `$sslst_data_root/kmeans_model/[SSL_MODEL]-[KM_TAG][PERCENTAGE]p-L[LAYER]-km[N_CLUSTER].bin`, e.g. `data/kmeans_model/hubert-ls0.01p-L9-km500.bin`.
4. Apply K-means model to SSL features by `bash script/s2u/apply_kmeans_simple.sh [DATASET] [SSL_MODEL] [LAYER] [N_CLUSTER] [KM_TAG]`. The results could be found as `$sslst_data_root/[DATASET]/[SPLIT].[SSL_MODEL]_l[LAYER]_[KM_TAG][N_CLUSTER]`, e.g. `data/libritrans-en-fr/dev.hubert_l9_ls0.01p500`
5. (Optional) Do the reduction by `bash script/s2u/reduce_hidden_unit.sh [DATASET] [SUFFIX] [MODE]`.
    
    If `mode == simple`, simply combine the consecutive characters. (E.g. aaabb -> ab)
    
    If `mode == addN`, add the number of consecutive after the character. (E.g. aaabb -> a _3 b _2)

## Training

## Inference