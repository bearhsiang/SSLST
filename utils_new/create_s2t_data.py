import argparse
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
import csv

# mandatory columns
KEY_ID, KEY_AUDIO, KEY_N_FRAMES = "id", "audio", "n_frames"
KEY_TGT_TEXT = "tgt_text"
# optional columns
KEY_SPEAKER, KEY_SRC_TEXT = "speaker", "src_text"
KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
# default values
DEFAULT_SPEAKER = DEFAULT_SRC_TEXT = DEFAULT_LANG = ""

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix')
    parser.add_argument('--src-lang')
    parser.add_argument('--tgt-lang')
    parser.add_argument('--manifest')
    parser.add_argument('--feat-root')
    parser.add_argument('--split')
    parser.add_argument('--output')
    args = parser.parse_args()

    return args

def main(args):

    df = pd.DataFrame()
    config = {}
    with open(args.manifest, 'r') as f:
        audio_root = f.readline().strip()
        audio_files, n_frames = [], []
        for line in f:
            audio_file, n_frame = line.strip().split('\t')
            audio_files.append(audio_file)
            n_frames.append(n_frame)
    
    if args.feat_root:
        audio_root = Path(args.feat_root)
        feat_files, n_frames = [], []
        for audio_file in tqdm(audio_files):
            feat_file = f'{args.split}/{Path(audio_file).stem}.npy'
            feat = np.load(audio_root/feat_file)
            n_frame = feat.shape[0]
            feat_files.append(feat_file)
            n_frames.append(n_frame)
        
        # rewrite audio_files with feat_files
        audio_files = feat_files

    config['audio_root'] = audio_root
    df[KEY_AUDIO] = audio_files
    df[KEY_N_FRAMES] = n_frames

    if args.src_lang:
        df[KEY_SRC_TEXT] = [line.strip() for line in open(f'{args.prefix}.{args.src_lang}')]
    if args.tgt_lang:
        df[KEY_TGT_TEXT] = [line.strip() for line in open(f'{args.prefix}.{args.tgt_lang}')]


    df.to_csv(
        args.output, 
        sep="\t",
        quotechar=None,
        doublequote=False,
        index_label=KEY_ID,
        quoting=csv.QUOTE_NONE,
    )

if __name__ == '__main__':

    args = get_args()
    main(args)
