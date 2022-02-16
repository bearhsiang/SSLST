import argparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm.auto import tqdm
import torch
from s2u_utils import FeatureExtractor, AudioDataset
from torch.utils.data import DataLoader
import random
from joblib import dump

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--upstream')
    parser.add_argument('-i', '--input-tsv')
    parser.add_argument('-k', '--audio-key')
    parser.add_argument('-d', '--audio-dir')
    parser.add_argument('-n', '--n_clusters', type=int, default=50)
    parser.add_argument('-s', '--seed', type=int, default=24)
    parser.add_argument('-l', '--layer', type=int)
    parser.add_argument('-o', '--output')
    parser.add_argument('-c', '--cuda', action='store_true')
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-f', '--frac', type=float)
    args = parser.parse_args()

    return args

def main(args):

    random.seed(args.seed)

    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    dataset = AudioDataset.from_tsv(args.input_tsv, args.audio_key, args.audio_dir, frac=args.frac)
    extractor = FeatureExtractor(args.upstream, device)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
    )

    all_features = []

    for batch in tqdm(dataloader):

        wavs = batch['wavs']

        try:
            wavs = [wav.to(device) for wav in wavs]
            features = extractor(wavs)[args.layer]
            features = [feature.cpu() for feature in features]
        
        except:
            print('out of memory, skip this batch')
            continue

            print('out of memory, using cpu')
            wavs = [wav.cpu() for wav in wavs]
            extractor.to('cpu')
            features = extractor(wavs)[args.layer]
            extractor.to(device)

        all_features += features

    all_features = torch.cat(all_features, dim=0)
    print(all_features.size())

    kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=args.seed)
    kmeans.fit(all_features)

    dump(kmeans, args.output)
    

if __name__ == '__main__':

    args = get_args()
    main(args)