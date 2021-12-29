#!/usr/local/bin bash

# config

input_root=data/tsv/
output_root=data/s2t/

name=covost2-en-de

### 

input_dir=$input_root/$name
output_dir=$output_root/$name/raw

python convert/s2t.py \
    -d $input_dir \
    -o $output_dir \
    -a path \
    -t translation \
    -s sentence \
    --speaker-key client_id