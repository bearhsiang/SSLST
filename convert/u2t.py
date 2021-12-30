import argparse
import pandas as pd
import csv

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--hidden-unit-file', required=True)
    parser.add_argument('-i', '--input-tsv', required=True)
    parser.add_argument('-o', '--output-tsv', required=True)
    parser.add_argument('-k', '--key', default='id')
    args = parser.parse_args()
    return args

def read_hidden_unit_file(file):

    data = {'index': [], 'hidden_unit':[]}
    with open(file, 'r') as f:
        for line in f:
            index, hidden_unit = line.strip().split('|')
            data['index'].append(index)
            data['hidden_unit'].append(hidden_unit)
    data = pd.DataFrame(data)
    return data

def main(args):

    data = pd.read_csv(
        args.input_tsv,
        delimiter='\t',
        quoting=csv.QUOTE_NONE,
    )
    hidden_unit_data = read_hidden_unit_file(args.hidden_unit_file)
    merged = data.set_index(args.key).join(
        hidden_unit_data.set_index('index')
    )

    merged.to_csv(
        args.output_tsv, 
        sep='\t',
        quoting=csv.QUOTE_NONE,
    )
    

if __name__ == '__main__':

    args = get_args()
    main(args)