import argparse
import torch
from s2u_utils import FeatureExtractor, AudioDataset
from torch.utils.data import DataLoader

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--upstream')
    parser.add_argument('-i', '--input-tsv')
    parser.add_argument('-k', '--audio-key')
    parser.add_argument('-d', '--audio-dir')
    parser.add_argument('-s', '--seed', type=int, default=24)
    parser.add_argument('-l', '--layer', type=int)
    parser.add_argument('-o', '--output')
    parser.add_argument('-c', '--cuda', action='store_true')
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    args = parser.parse_args()

    return args

def main(args):

    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    dataset = AudioDataset.from_tsv(args.input_tsv, args.audio_key, args.audio_dir)
    extractor = FeatureExtractor(args.upstream, device)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=True,
    )



if __name__ == '__main__':

    args = get_args()
    main(args)