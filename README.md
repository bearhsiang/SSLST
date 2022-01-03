# SSLST

## Modules

### prepare_data

convert the public datasets to unified tsv format, each subset (e.g. train, dev, test, ...) belongs to different tsv files.

```
+-- data
    +-- covost2-en-de
        +-- raw
            +-- train.tsv
            +-- dev.tsv
            +-- test.tsv
        +-- normalized
            +-- ...
    +-- MustC-en-de
        +-- raw
            +-- train.tsv
            +-- dev.tsv
            +-- tst-COMMON.tsv
            +-- tst-HE.tsv
```

### convert

convert the unified tsv format to official toolkit input format. (e.g. fairseq, ...)

#### s2t

Convert the tsv file into fariseq Speech to Text format with required fields: `id`, `audio`, `n_frames`, `tgt_text`

Reference: [fairseq/data/audio/speech_to_text_dataset.py](https://github.com/pytorch/fairseq/blob/ee177fc4fa06dcb3d5fd466559af1b46893c00e8/fairseq/data/audio/speech_to_text_dataset.py#L384-L392)

### TODOs

- [ ] n_frames in convert/s2t.py