import argparse
from pathlib import Path
import s3prl.hub as hub
import soundfile as sf
from tqdm.auto import tqdm
import torch
from npy_append_array import NpyAppendArray
import logging
import os
import sys

S3PRL_SR = 16000

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
    parser.add_argument('--format', choices=['seperate', 'collect'])
    args = parser.parse_args()

    return args

def get_features(model, layer, device, audio_dir, audio_files):

    for audio_file, n_frames in tqdm(audio_files):

        wav, sr = sf.read(audio_dir/audio_file)
        assert wav.shape[0] == int(n_frames)
        assert sr == S3PRL_SR

        with torch.no_grad():

            feat = model([torch.tensor(wav, dtype=torch.float).to(device)])
            feat = feat['hidden_states'][layer][0]

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

        for feat in get_features(model, args.layer, args.device, audio_dir, audio_files):

            feat_f.append(feat.cpu().numpy())
            print(feat.size(0), file=len_f)
        
        len_f.close()

    elif args.format == 'seperate':

        raise NotImplementedError


if __name__ == '__main__':
    
    args = get_args()
    logger.info(args)
    main(args)