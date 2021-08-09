import csv

def read_tsv(file):
    lines = []
    with open(file, 'r') as f:
        reader = csv.DictReader(
            f,
            delimiter='\t',
            quotechar=None,
            doublequote=False,
            lineterminator='\n',
            quoting=csv.QUOTE_NONE,
        )
        for line in reader:
            lines.append(line)
    return lines

def write_tsv(file, data, header=None):

    if not header:
        header = list(data[0].keys())

    with open(file, 'w') as f:
        writer = csv.DictWriter(
            f,
            delimiter='\t',
            quotechar=None,
            doublequote=False,
            lineterminator='\n',
            quoting=csv.QUOTE_NONE,
            fieldnames=header,
        )
        writer.writeheader()
        writer.writerows(data)

# from fairseq
def is_npy_data(data: bytes):
    return data[0] == 147 and data[1] == 78