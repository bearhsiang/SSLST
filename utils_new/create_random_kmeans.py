import argparse
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import joblib

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dim', type=int, required=True)
    parser.add_argument('-n', '--n-clusters', type=int, required=True)
    parser.add_argument('-m', '--mean', type=float, required=True)
    parser.add_argument('-s', '--std', type=float, required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    return args

def main(args):
    
    if args.seed:
        np.random.seed(args.seed)

    km_model = MiniBatchKMeans(n_clusters=args.n_clusters)
    km_model.cluster_centers_ = np.random.normal(args.mean, args.std, size=(args.n_clusters, args.dim))

    joblib.dump(km_model, args.output)


if __name__ == '__main__':

    args = get_args()
    main(args)