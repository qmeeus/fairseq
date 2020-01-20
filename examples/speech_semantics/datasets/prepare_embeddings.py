#!/usr/bin/env python


from gensim.models import KeyedVectors
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser("Convert texts to embeddings")
    parser.add_argument("--data", type=str, help="Path to the texts")
    parser.add_argument("--embeddings", type=str, help="Path to the embeddings")
    return parser.parse_args()


def process_texts(textfile):
    with open(textfile) as f:
        lines = f.readlines()




def main():

    args = parse_args()

    # Load the embeddings
    model = KeyedVectors.load_word2vec_format(args.embeddings, binary=False)



