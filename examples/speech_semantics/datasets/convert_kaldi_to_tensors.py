#!/usr/bin/env python

# Not working, use convert_wav_to_tensors.py

import os
import os.path as p
import re
import argparse
import torch
import torchaudio

# /esat/spchdisk/scratch/jponcele/KALDI23/egs/CGN/data/dev_s/data


def parse_args():
    parser = argparse.ArgumentParser(
        "Converts kaldi-style ark files to torch tensors")
    parser.add_argument("--kaldi_dir", type=str,
                        help="Location of kaldi files")
    parser.add_argument("--dest_dir", type=str,
                        help="Where to save the data")
    return parser.parse_args()


def is_ext(extension):
    def _check(filename):
        return filename.endswith("scp")
    return _check

def main():
    args = parse_args()
    os.makedirs(args.dest_dir, exist_ok=True)
    source_files = list(filter(is_ext("scp"), os.listdir(args.kaldi_dir)))
    for filename in source_files:
        with open(p.join(args.kaldi_dir, filename), "rb") as file:
            d = { u: d for u, d in torchaudio.kaldi_io.read_vec_flt_scp(file) }
            print(len(d))
            print(d.keys())
            return
            # name, tensor = torchaudio.kaldi_io.read_vec_int_ark(ark_file)
        print(name, type(tensor))
        save_path = p.join(args.dest_dir, re.sub("\.scp^", ".pt", filename))
        torch.save(tensor, save_path)


if __name__ == '__main__':
    main()
