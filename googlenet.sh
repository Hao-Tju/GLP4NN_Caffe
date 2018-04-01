#!/usr/bin/env bash

set -e

if [ -f ./Makefile.config.prof ]; then
  echo "Now is doing regular caffe training process ..."

  make clean
  make -j 8

  ./models/bvlc_googlenet/train_googlenet.sh

  mv Makefile.config Makefile.config.serial
  mv Makefile.config.prof Makefile.config

  echo "Now is doing GLP4NN caffe training process ..."

  make clean
  make -j 8

  ./models/bvlc_googlenet/train_googlenet_cpnn.sh

  exit
fi

if [ -f ./Makefile.config.serial ]; then
  echo "Now is doing GLP4NN caffe training process ..."

  make clean
  make -j 8

  ./models/bvlc_googlenet/train_googlenet_cpnn.sh

  mv Makefile.config Makefile.config.prof
  mv Makefile.config.serial Makefile.config

  echo "Now is doing regular caffe training process ..."

  make clean
  make -j 8

  ./models/bvlc_googlenet/train_googlenet.sh

  exit
fi
