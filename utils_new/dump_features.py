import argparse
from pathlib import Path
import s3prl.hub as hub
from tqdm.auto import tqdm
import torch
from npy_append_array import NpyAppendArray
import logging
import os
import sys
import numpy as np
import torchaudio

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_features")

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-dir')
    parser.add_argument('--split')
    parser.add_argument('--model', choices=[model for model in dir(hub) if model[0] != '_'])
    parser.add_argument('--layer', type=int)
    parser.add_argument('--nshard', type=int)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir')
    parser.add_argument('--format', choices=['seperate', 'collect'], default='collect')
    parser.add_argument('--sample-rate', type=int, default=16000)
    args = parser.parse_args()

    return args

def get_features(model, layer, device, audio_dir, audio_files, sample_rate, get_path=False):

    resampler = {}

    for audio_file, n_frames in tqdm(audio_files):

        wav, sr = torchaudio.load(audio_dir/audio_file)

        if sr != sample_rate:
            if sr not in resampler:
                resampler[sr] = torchaudio.transforms.Resample(sr, sample_rate)
            wav = resampler[sr](wav)
            sr = sample_rate

        wav = wav.mean(0)

        assert len(wav) == int(n_frames)
        assert sr == sample_rate

        with torch.no_grad():

            feat = model([torch.tensor(wav, dtype=torch.float).to(device)])
            feat = feat['hidden_states'][layer][0]

        if get_path:
            yield feat, audio_dir / audio_file
        else:
            yield feat

def main(args):

    manifest_file = Path(args.manifest_dir)/f'{args.split}.tsv'

    with open(manifest_file) as f:
        audio_dir = Path(f.readline().strip())
        audio_files = [line.strip().split('\t') for line in f]

    begin = round(len(audio_files)*args.rank/args.nshard)
    end = round(len(audio_files)*(args.rank+1)/args.nshard)
    audio_files = audio_files[begin:end]

    model = getattr(hub, args.model)()
    model.to(args.device)
    model.eval()

    with torch.no_grad():

        fake_wav = [torch.randn(160000, dtype=torch.float).to(args.device)]
        fake_feat = model(fake_wav)['hidden_states']
        n_layers = len(fake_feat)

    logger.info(f'select layer: {args.layer}/(0~{n_layers-1})')

    if args.layer < 0 or args.layer >= n_layers:
        logger.error(f'invalid layer: {args.layer} (0~{n_layers-1})')

        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.format == 'collect':

        output_feat = output_dir/f'{args.split}_{args.rank}_{args.nshard}.npy'
        output_len = output_dir/f'{args.split}_{args.rank}_{args.nshard}.len'

        feat_f = NpyAppendArray(output_feat)
        len_f = open(output_len, 'w')

        for feat in get_features(model, args.layer, args.device, audio_dir, audio_files, args.sample_rate):

            feat_f.append(feat.cpu().numpy())
            print(feat.size(0), file=len_f)
        
        len_f.close()

    elif args.format == 'seperate':

        output_dir = output_dir / args.split
        output_dir.mkdir(parents=True, exist_ok=True)

        for feat, file_path in get_features(model, args.layer, args.device, audio_dir, audio_files, args.sample_rate, get_path=True):

            output_path = output_dir / f'{file_path.stem}.npy'
            np.save(output_path, feat.cpu().numpy())


if __name__ == '__main__':
    
    args = get_args()
    logger.info(args)
    main(args)