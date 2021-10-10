import argparse
import json
import numpy as np
from numpy.linalg import norm
from scipy import stats
from tqdm.auto import tqdm
from collections import defaultdict

def count_sim(a, b):

    return np.inner(a, b)/(norm(a)*norm(b))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True)
    parser.add_argument('--simfile', required=True)
    args = parser.parse_args()

    with open(args.json) as f:
        w2feat = json.load(f)


    feat_score = defaultdict(list)
    true_score = []

    count = 0

    lines = []

    for line in open(args.simfile):
        lines.append(line)

    for line in tqdm(lines):

        line = line.strip().split('\t')
        w1, w2, score = line[0], line[1], line[3]

        count += 1

        if w1 in w2feat and w2 in w2feat:

            assert len(w2feat[w1]) == len(w2feat[w2])

            for layer in range(len(w2feat[w1])):

                sim = count_sim(w2feat[w1][layer], w2feat[w2][layer])
                feat_score[layer].append(sim)
            
            true_score.append(score)

    print(f'{len(true_score)}/{count}')

    for layer in feat_score:
        print(layer, stats.spearmanr(feat_score[layer], true_score))
    
