import json
from argparse import ArgumentParser
from os import path 

"""
This script creates a new annotations file for the COCO data, including only
those entries for which a corresponding NPY file was found.

This is for debug on the COCO+depth data, since it may take many hours to create
pseudo-depth images for all COCO files
"""

parser = ArgumentParser()
parser.add_argument('train_root', type=str)
parser.add_argument('in_annot', type=str)
parser.add_argument('out_annot', type=str)
args = parser.parse_args()
with open(args.in_annot, 'r') as r:
    d = json.load(r)
    d['images'] = [x for x in d['images'] if path.exists(path.join(args.train_root, x['file_name'].replace('.jpg', '.npy')))]
    print(d['images'])
    print(len(d['images']))

    with open(args.out_annot, 'w+') as w:
        json.dump(d, w)