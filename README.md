# SSLST

This repo is for utilizing Self-Supervised Learning (SSL) speech models in speech translation. It supports three tasks relavant to speech translation:
1. **T2T**: Machine Translation for transcription to translation.
2. **S2T**: Speech Translation for speech SSL features to translation.
3. **U2T**: Speech Translation for speech SSL discrete units to translation.

## Setup

Please first setup the following configuration in `script/setup.sh`. (You could do this by copying and modifying `script/setup_example.sh`)
* `sslst_data_root`: The place to put the processed data.
* `sslst_feat_root`: The place to store the extracted features. (A large storage would be needed.)
* `sslst_output_root`: The place to store the checkpoints.
* `sslst_data_bin_root`: The place to store binarized text data.

## Get data

The speech dataset will be processed into tsv files. This repo default supports three common datasets, which are librispeech, libritrans and covost2, and also allow you to add new dataset.

### To prepare supported dataset

* Librispeech
    1. Download Librispeech from [Official website](https://www.openslr.org/12).
    We set `train-clean-100`, `dev-clean`, `test-clean` as default splits, modifying `prepare_data/Librispeech.py` to change this setting.
    2. Set `$sslst_librispeech_root` in `script/setup.sh`
    3. Run `bash prepare_data/librispeech.sh`


* Libritrans
    1. Download Libritrans from [this repo](https://github.com/alicank/Translation-Augmented-LibriSpeech-Corpus)
    2. Set `$sslst_libritrans_root` in `script/setup.sh`
    3. Run `bash prepare_data/libritrans.sh`

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
3. Use `script/t2t/fairseq_preprocess.sh` to binarize the data.
    ```bash
    bash script/t2t/fairseq_preprocess.sh [DATASET] [SRC_LANG] [TGT_LANG]
    ```

### Speech

#### Speech to hidden unit

1. Create manifest by `bash script/s2u/create_manifest_[DATASET].sh`
2. Clone and install [Fairseq](https://github.com/facebookresearch/fairseq). Set `$sslst_fairseq_root` in `script/setup.sh`
3. Train K-means model.
    ```bash
    bash script/s2u/train_kmeans_simple.sh [DATASET] [KM_TAG] [SSL_MODEL] [LAYER] [N_CLUSTER] [PERCENTAGE]
    ``` 
    If you didn't use `librispeech`, you need to change `split` from `train-clean-100` into other one in `script/s2u/train_kmeans_simple.sh`.

    The kmeans model could be found as `$sslst_data_root/kmeans_model/[SSL_MODEL]-[KM_TAG][PERCENTAGE]p-L[LAYER]-km[N_CLUSTER].bin`, e.g. `data/kmeans_model/hubert-ls0.01p-L9-km500.bin`.

4. Dump SSL features and apply K-means clustering,
    ```bash
    bash script/s2u/apply_kmeans_simple.sh [DATASET] [SSL_MODEL] [LAYER] [N_CLUSTER] [KM_TAG]
    ```

    The results could be found as `$sslst_data_root/[DATASET]/[SPLIT].[SSL_MODEL]_l[LAYER]_[KM_TAG][N_CLUSTER]`, e.g. `data/libritrans-en-fr/dev.hubert_l9_ls0.01p500`.
    
    The dump SSL features are in `$sslst_feat_root/[DATASET]/[SSL_MODELS]/[LAYER]`

5. (Optional) Do the reduction.
    ```bash
    bash script/s2u/reduce_hidden_unit.sh [DATASET] [SUFFIX] [MODE]
    ```
    
    If `mode == simple`, it simply combines the consecutive characters. (E.g. aaabb -> ab)
    
    If `mode == addN`, it will add the number of consecutive after the character. (E.g. aaabb -> a _3 b _2)

6. Use `script/t2t/fairseq_preprocess.sh` to binarize the data.
    ```bash
    bash script/t2t/fairseq_preprocess.sh [DATASET] [SUFFIX] [TGT_LANG]
    ```

#### Speech to text

1. Dump the ssl features.
    ```bash 
    bash script/dump_feature.sh [DATASET] [SSL_MODEL] [LAYER] seperate
    ```

    If you have already ran the speech to hidden unit script, you could simply split those features.
    ```bash
    bash script/s2t/split_feature.sh [DATASET] [SSL_MODEL] [LAYER]
    ```

2. Create Speech-to-Text task configuration
    ```bash
    bash script/s2t/speech2text.sh [DATASET] [SSL_MODEL] [SSL_DIM] [LAYER] [SRC_LANG] [TGT_LANG]
    ```

## Training

### Text-to-text
We use Fairseq's [transformer_iwslt_de_en](https://github.com/facebookresearch/fairseq/blob/b5e7b250913120409b872a940fbafec4d43c7b13/fairseq/models/transformer/transformer_legacy.py#L224) as our default model architecture.
```bash
bash script/train/translation_t2t.sh [DATASET] [SRC_LANG] [TGT_LANG]
```

### Unit-to-text
We also use Fairseq's [transformer_iwslt_de_en](https://github.com/facebookresearch/fairseq/blob/b5e7b250913120409b872a940fbafec4d43c7b13/fairseq/models/transformer/transformer_legacy.py#L224) as our default model architecture. 
```bash
bash script/train/translation_u2t.sh [DATASET] [SUFFIX] [TGT_LANG]
```

### Speech-to-text 
We use Fairseq's [s2t_transformer_s](https://github.com/facebookresearch/fairseq/blob/b5e7b250913120409b872a940fbafec4d43c7b13/fairseq/models/speech_to_text/s2t_transformer.py#L515) as our default model architecture.
```bash
bash script/train/speech2text.sh [DATASET] [SRC_LANG] [TGT_LANG]
```

The detail of the hyperparameters could be found in the training scripts.

## Inference

We report SacreBLEU as performance metric.

### Cascade system

* speech-to-text-to-text
    ```bash 
    bash script/generate/cascade_s2t2t.sh [DATASET] [SRC_LANG] [MID_LANG] [TGT_LANG]
    ```
* Unit-to-text-to-text
    ```bash 
    bash script/generate/cascade_u2t2t.sh [DATASET] [SRC_LANG] [MID_LANG] [TGT_LANG]
    ```

### End-to-end system

* Text-to-text
    ```bash 
    bash script/generate/translation_t2t.sh [DATASET] [SRC_LANG] [TGT_LANG]
    ```
* Unit-to-text
    ```bash 
    bash script/generate/translation_u2t.sh [DATASET] [SRC_LANG] [TGT_LANG]
    ```
* Speech-to-text
    ```bash 
    bash script/generate/speech2text.sh [DATASET] [SRC_LANG] [TGT_LANG]
    ```

## Finetune from mBART

### Preprocess

1. Download `mbart.cc25` from [fairseq](https://github.com/facebookresearch/fairseq/blob/main/examples/mbart/README.md).
2. Unzip and put the folder into `$sslst_data_root`

### Tokenized text with mBART's spm model

1. Set the dataset and langauge in `script/finetune/mbart-t2t.sh` properly.

2. Run the script to create the binarized dataset.
    ```bash
    bash script/finetune/mbart-t2t.sh
    ```

### Convert hidden unit into mbart's subword

1. Create the mapping of hidden units and subwords
    ```bash
    bash script/finetune/mbart-create_hidden_unit_mapping.sh [DATASET] [LANG]
    ```

2. Apply the mapping and create binarized dataset
    ```bash
    bash script/finetune/mbart-u2t.sh [DATASET] [SRC_LANG] [MBART_TGT_LANG]
    ```

### Training

#### Text-to-text

```bash
bash script/train/translation_t2t_mbart.sh [DATASET] [SRC_LANG] [TGT_LANG]
```

#### Unit-to-text

```bash
bash script/train/translation_u2t_mbart.sh [DATASET] [SRC_LANG] [TGT_LANG]
```

### Inference

#### Text-to-text
```bash
bash script/generate/translation_t2t_mbart.sh [DATASET] [SRC_LANG] [TGT_LANG]
```

#### Unit-to-text
```bash
bash script/generate/translation_u2t_mbart.sh [DATASET] [SRC_LANG] [TGT_LANG]
```