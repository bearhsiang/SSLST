# SSLST

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

The data would be prepared at `$sslst_data_root/tsv` in tsv format.