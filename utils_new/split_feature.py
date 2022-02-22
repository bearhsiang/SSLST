import argparse
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--feat-dir')
    parser.add_argument('--split')
    parser.add_argument('--nshard', type=int)
    parser.add_argument('--manifest')
    parser.add_argument('--out-feat-dir')
    args = parser.parse_args()
    
    return args

def main(args):

    with open(args.manifest, 'r') as f:
        audio_root = f.readline().strip()
        audio_files = [ line.split('\t')[0] for line in f ]

    feat_dir = Path(args.feat_dir)
    lens_list = [ [int(line.strip()) for line in open(feat_dir/f'{args.split}_{rank}_{args.nshard}.len')] for rank in range(args.nshard)]

    assert len(audio_files) == sum([len(lens) for lens in lens_list]), f'files in manifest: {len(audio_files)} is not as same as in feature directory: {sum([len(lens) for lens in lens_list])}'

    audio_files_offset = 0
    out_feat_dir = Path(args.out_feat_dir)
    out_feat_dir.mkdir(parents=True, exist_ok=True)

    for rank in range(args.nshard):

        feat = np.load(feat_dir/f'{args.split}_{rank}_{args.nshard}.npy', mmap_mode="r")
        assert feat.shape[0] == sum(lens_list[rank])

        length_offset = 0

        for i in tqdm(range(len(lens_list[rank]))):
            
            length = lens_list[rank][i]
            audio_file = audio_files[audio_files_offset+i]
            audio_feat = feat[length_offset: length_offset+length]

            np.save(out_feat_dir/f'{Path(audio_file).stem}.npy', audio_feat)

            length_offset += length
        
        audio_files_offset += len(lens_list[rank])
    
    assert audio_files_offset == len(audio_files)


if __name__ == '__main__':

    args = get_args()
    print(args)
    main(args)