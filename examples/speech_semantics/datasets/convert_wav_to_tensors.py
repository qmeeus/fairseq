#!/usr/bin/env python

import os
from pathlib import Path
import re
import argparse
import torch
import torchaudio
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        "Converts raw audio files to torch tensors")
    parser.add_argument("--filelist", type=Path,
                        help="Textfile with all the paths to the audiofiles")
    parser.add_argument("--dest-dir", type=Path,
                        help="Where to save the data")
    parser.add_argument("--features", type=str, default="MFCC", 
                        help="The kind of features, one of MFCC or logmels")
    parser.add_argument("--n-features", type=int, default=40, 
                        help="Number of mel filterbanks")
    parser.add_argument("--max-sequence-length", type=int, default=0, 
                        help="Pad or truncate sequence (0 = no padding)")
    return parser.parse_args()


def parse_pathfile(pathfile):
    with open(pathfile) as f:
        path_str = filter(bool, map(str.strip, f.readlines()))
        paths = list(Path, path_str)
        assert all(map(Path.exists, paths))
        return paths


def main():
    args = parse_args()
    os.makedirs(args.dest_dir, exist_ok=True)

    transformer = None

    with open(args.filelist) as f:
        audiofiles = list(map(Path, filter(bool, map(str.strip, f.readlines()))))

    for audiofile in audiofiles:
        waveform, sample_rate = torchaudio.load(audiofile)

        # print("Shape of waveform: {}".format(waveform.size()))
        # print("Sample rate of waveform: {}".format(sample_rate))
        # plt.figure()
        # plt.plot(waveform.t().numpy())
        # plt.show()

        if transformer is None:
            if args.features == "MFCC":
                transformer = torchaudio.transforms.MFCC(
                    sample_rate=sample_rate,
                    n_mfcc=args.n_features)
            else:
                transformer = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_mels=args.n_features
                )

        features = transformer(waveform)

        # print("Shape of features: {}".format(features.size()))
        # plt.figure()
        # p = plt.imshow(features.log2()[0,:,:].detach().numpy(), aspect='auto')
        # plt.show()

        if args.max_sequence_length > 0:
            features = F.pad(features, (0, 0, args.max_sequence_length - features.size(-1)))
        
        output_file = args.dest_dir.joinpath(*audiofile.parts[1:-1], audiofile.name.replace(".wav", ".pt"))
        os.makedirs(output_file.parent, exist_ok=True)
        torch.save(features, output_file)


if __name__ == '__main__':
    main()