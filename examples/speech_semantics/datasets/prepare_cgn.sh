#!/bin/bash -xe


DATASETS_LOCATION=${1%/}
EMBEDDINGS_DIR=${3%/}

EMBEDDINGS_URL=http://www.clips.uantwerpen.be/dutchembeddings/combined-160.tar.gz

echo "Downloading the embeddings"
mkdir -p $EMBEDDINGS_DIR && cd $EMBEDDINGS_DIR
wget -O- $EMBEDDINGS_URL | tar -xf

echo "Preparing the embeddings"
python prepare_embeddings.py --data $DATASETS_LOCATION/texts --embeddings $EMBEDDINGS_DIR

