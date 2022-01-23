import argparse
from s2u_utils import FeatureExtractor, AudioDataset
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import joblib
from pathlib import Path

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--upstream', required=True)
    parser.add_argument('-i', '--input-tsv', required=True)
    parser.add_argument('-k', '--audio-key', required=True)
    parser.add_argument('-d', '--audio-dir', default='.')
    parser.add_argument('-l', '--layer', type=int, default=-1)
    parser.add_argument('--kmeans-model', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default = 1)
    args = parser.parse_args()
    return args

def main(args):
    
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    dataset = AudioDataset.from_tsv(args.input_tsv, args.audio_key, args.audio_dir)
    extractor = FeatureExtractor(args.upstream, device)
    km_model = joblib.load(args.kmeans_model)
    km_model.verbose = 0

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
    )

    output = Path(args.output)
    output.parent.mkdir(exist_ok=True, parents=True)

    with open(output, 'w') as f:
        for batch in tqdm(dataloader):
            names, wavs = batch['names'], batch['wavs']
            try:
                wavs = [wav.to(device) for wav in wavs]
                features = extractor(wavs)[args.layer]
                features = [feature.cpu() for feature in features]
            
            except:
                print('out of memory, using cpu')
                wavs = [wav.cpu() for wav in wavs]
                features = [feature.cpu() for feature in features]
                extractor = extractor.to('cpu')
                features = extractor(wavs)[args.layer]
                extractor = extractor.to(device)

            for name, feature in zip(names, features):
                feature = feature.numpy()
                unit = km_model.predict(feature).tolist()
                print(name.stem, " ".join([str(i) for i in unit]), sep='|', file=f)



if __name__ == '__main__':
    args = get_args()
    main(args)