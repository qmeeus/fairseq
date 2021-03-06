#!/usr/bin/env python

import os
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
import gzip
import argparse
import torch
import torchaudio
import torch.functional as F

from logger import logger

"""
Prepare the CGN dataset and extract spoken sentences representations (either the raw waveform, the logmel 
or the MFCC) with their written transcription. The script expects a CSV file as input that contains the
paths to the audio recordings (WAV) and orthographic transcriptions (XML).
"""



def parse_args():
    parser = argparse.ArgumentParser("Converts raw audio files to torch tensors")
    parser.add_argument("--input-file", type=Path,
                        help="File that contains all the paths to the audiofiles and transcriptions")
    parser.add_argument("--data-dir", type=Path, 
                        help="Path from which the paths in input-file descend")
    parser.add_argument("--dest-dir", type=Path,
                        help="Where to save the data")
    parser.add_argument("--features", type=str, default="MFCC", 
                        help="The kind of features, one of MFCC or logmels")
    parser.add_argument("--n-features", type=int, default=40, 
                        help="Number of mel filterbanks")
    parser.add_argument("--max-sequence-length", type=int, default=0, 
                        help="Pad or truncate sequence (0 = no padding)")
    return parser.parse_args()


def common_path(path_a, path_b):
    """
    Compute the deepest common path between path_a and path_b
    :path_a,path_b: pathlib.PurePath objects
    returns pathlib.PurePath object that represent the deepest common subpath 
    """ 
    return Path(*os.path.commonprefix([path_a.parts, path_b.parts]))


def spoken_sentence_generator(audiofile, textfile,
                              min_sequence_length=0,
                              max_sequence_length=0, 
                              feature_type="MFCC", 
                              n_features=40, 
                              **options):
    """
    Generator to extract sentences from a text transcription of audio file and spoken sentences 
    features from the corresponding audio file. The features can be one of None (raw waveform), 
    MFCC or logmel. 
    :textfile: the path to the text file
    :audiofile: the path to the audio file
    :min_sequence_length: int, minimim number of millisecond for an utterance to be valid 
    :max_sequence_length: int, the length of the sequence
    :Preprocessor: class from torchaudio.transform
    :preprocessor_options: dict with options to initialise the preprocessor
    returns a generator of tuples (sentence_id features, speaker, sentence) where features is a 
    torch.Tensor, speaker is a string identifier and sentence is a text.
    """

    # logger.debug(f"Loading {textfile}")
    with gzip.open(textfile, encoding='latin-1', mode='rt') as f:
        tree = BeautifulSoup(f, "lxml")

    # Load the first channel of the wav file
    waveform, sample_rate = torchaudio.load(audiofile)
    waveform = waveform[None, :1, :]
    logger.debug(f"{audiofile} loaded: size: {tuple(waveform.size())} rate: {sample_rate}")

    options.update({"sample_rate": sample_rate})
    if feature_type.lower() == "mfcc":
        options.update({"n_mfcc": n_features})
        f = torchaudio.transforms.MFCC(**options)
    elif feature_type == "logmel":
        options = {"n_mels": n_features}
        f = torchaudio.transforms.MelSpectrogram(**options)
    else:
        f = None

    sentence_id = 0
    for utterance in tree.find_all("tau"):

        speaker = utterance.get("s")
        start = float(utterance.get("tb"))
        end = float(utterance.get("te"))

        if (end - start) * 1000 < min_sequence_length:
            logger.debug(f"Ignoring utterance {sentence_id} (length: {end-start:.2f})")
            continue

        sentence = " ".join([word.get("w") for word in utterance.find_all("tw")])
        logger.debug(f"Found utterance from {start} to {end}: {sentence}")

        start, end = (int(t * sample_rate) for t in (start, end))
        feats = waveform[:, :, start:end]
        logger.debug(f"Truncating signal from {start} to {end}, size: {tuple(feats.size())}")

        if f is not None:
            feats = f(feats)

        yield sentence_id, feats, speaker, sentence
        sentence_id += 1


def pad_sequence(seq, maxlen): 
    """
    Pad / truncate the last axis of a sequence to the specified length
    :seq: torch.Tensor, the sequence to be padded
    :maxlen: int, the length of the sequence
    returns torch.Tensor, the padded sequence  
    """
    padding = [0] * (seq.ndim - 1) + [maxlen - seq.size(-1)]
    return F.pad(seq, padding)


def filter_dataframe(dataframe, exclude=False, **filters): 
    """
    Apply filters to a pandas.DataFrame. Can either include or exclude the given values
    :dataframe: pd.DataFrame, data to be filtered
    :exclude: boolean, whether to include (default) or exclude the given values
    :filters: key-value pairs, key is the same of a column in the dataframe and value can be
        one value or a list of values to include/exclude
    returns a filtered dataframe 
    """
    
    for key, value in filters.items():

        if type(value) is list:
            mask = dataframe[key].isin(value)            
        else:
            mask = dataframe[key] == value
        if exclude:
            mask = ~mask
        dataframe = dataframe[mask]

    return dataframe
            

def generate_data_from_file(filename, root=None, include=None, exclude=None, **options):
    """
    Generator to bulk create the datasets from a CSV with 4 columns: 
        - comp (comp-[a-z]): the component to which the file exists (see CGN)
        - lang ([nl|vl]): the language of the recording
        - name (f[n|v]\d{6}): the identifier of the recording
        - audio (path-like): the path to the audio recording (wav)
        - text (path-like): the path to the orthographical retranscription (skp.gz)
    :filename: the path to the csv file
    :root: the path from which the filenames should be considered. (None = same dir as filename)
    :include,exclude: dict-like, key-value pairs to filter the csv (keys must be one of comp/name)
    :options: additional options to pass to spoken_sentence_generator
    returns a generator of tuples (comp, lang, name, sentence_id, features, speaker, sentence) where 
    features is a torch.Tensor, speaker is a string identifier and sentence is a text.
    """
    paths = pd.read_csv(filename)

    assert all(col in paths for col in ("comp", "lang", "name", "audio", "text")), "Invalid CSV"
    assert len(paths), "Empty CSV"

    logger.debug(f"Loaded {filename} with {len(paths)} target recordings")

    for flag, filters in enumerate([include, exclude]):
        if filters:
            paths = filter_dataframe(paths, exclude=flag, **filters)

    assert len(paths), "No more results, filters might be too stricts."
    logger.debug(f"{len(paths)} target files remaining after filtering")

    for _, comp, lang, name, audiofile, textfile in paths.itertuples():
        if root is not None:
            audiofile, textfile = (Path(root, fn) for fn in (audiofile, textfile))
        for retval in spoken_sentence_generator(audiofile, textfile, **options):
            yield tuple([comp, lang, name] + list(retval)) 


def main():

    args = parse_args()

    # OPTIONS
    # determines how files are found and which to include
    INPUT_FILE = args.input_file
    ROOT = args.data_dir
    INCLUDE_FILTERS = None
    EXCLUDE_FILTERS = None
    
    # where to save the files
    SAVE_DIRECTORY = args.dest_dir
    TEXT_OUTPUT_FILE = "sentences.txt"

    # to be passed to spoken_sentence_generator
    FEATURES_OPTS = {
        'min_sequence_length': 2000,
        'max_sequence_length': args.max_sequence_length,
        'feature_type': args.features,
        'n_features': args.n_features,
    }

    # Sanity check
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    
    with open(Path(SAVE_DIRECTORY, TEXT_OUTPUT_FILE), 'w') as txtfile:

        for comp, lang, name, sent_id, feats, spkr, sent in generate_data_from_file(
            INPUT_FILE, ROOT, INCLUDE_FILTERS, EXCLUDE_FILTERS, **FEATURES_OPTS):

            output_file = Path(comp, lang, f"{name}.{sent_id:06d}.pt")
            output_path = Path(SAVE_DIRECTORY, output_file)

            txtfile.write(f"{output_file}\t{sent}\n")
            os.makedirs(output_path.parent, exist_ok=True)
            torch.save(feats, output_path)
            txtfile.flush()


if __name__ == '__main__':
    main()
