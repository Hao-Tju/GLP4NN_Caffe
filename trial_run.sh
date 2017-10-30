#/bin/bash

set -e

if [ $# -lt 2 ]; then
  echo "WRONG Usage!"
  echo "USAGE: $0 <executable command configuration file>"
  exit
fi

if [ -f ./Makefile.config.prof ]; then
  echo "Now is doing regular caffe training process ..."
fi

if [ -f ./Makefile.config.serial ]; then
  echo "Now is doing parallelized caffe training process ..."
fi

# clean former executable files and object files.
make clean
make

while read -r line || [ -n "$line" ]; do
  $line
done < "$1"

# PROF || Serial test

if [ -f ./Makefile.config.prof ]; then
  mv ./Makefile.config ./Makefile.config.serial
  mv ./Makefile.config.prof ./Makefile.config
elif [ -f ./Makefile.config.serial ]; then
  mv ./Makefile.config ./Makefile.config.prof
  mv ./Makefile.config.serial ./Makefile.config
fi

make clean
make

while read -r line || [ -n "$line" ]; do
  $line
done < "$1"
