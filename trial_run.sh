#/bin/bash

set -e

if [ $# -lt 1 ]; then
  echo "WRONG USAGE! MISSING argument!"
  echo "USAGE: $0 <command text>"
  exit
fi

if [ -f ./Makefile.config.prof ]; then
  echo "Now is doing regular caffe training process ..."
fi

if [ -f ./Makefile.config.serial ]; then
  echo "Now is doing parallelized caffe training process ..."
fi

# clean former executable files and object files.
make

i=0
while read -r line || [ -n "$line" ]; do
  $line
  if [ -f ./Makefile.config.serial ] && [ -f ./LOG/CUPTI-OVERHEAD.csv ]; then
    echo $i
    mv ./LOG/CUPTI-OVERHEAD.csv ./LOG/CUPTI_OVERHEAD_$i.csv
    i=$(($i+1))
  fi
done < "$1"

# PROF || Serial test

if [ -f ./Makefile.config.prof ]; then
  mv ./Makefile.config ./Makefile.config.serial
  mv ./Makefile.config.prof ./Makefile.config
  echo "Now is doing parallel execution ..."
elif [ -f ./Makefile.config.serial ]; then
  mv ./Makefile.config ./Makefile.config.prof
  mv ./Makefile.config.serial ./Makefile.config
  echo "Now is doing serial execution ..."
fi

make clean
make

i=0
while read -r line || [ -n "$line" ]; do
  $line

  echo "Finished $line"
  if [ -f ./Makefile.config.serial ] && [ -f ./LOG/CUPTI-OVERHEAD.csv ]; then
    echo "[CURRENT CUPTI-OVERHEAD]: $i"
    mv ./LOG/CUPTI-OVERHEAD.csv ./LOG/CUPTI-OVERHEAD-$i.csv
    i=$(($i++))
  fi
done < "$1"
