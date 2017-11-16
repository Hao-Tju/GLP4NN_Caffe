#/bin/bash

set -e

echo "$@ $#"

exec_cmds="$1"
cpnn_exec_cmds="$2"
curr_cmds=""

if [ $# -lt 1 ]; then
  echo "WRONG USAGE! MISSING argument!"
  echo "USAGE: $0 <command text>"
  exit
fi

if [ -f ./Makefile.config.prof ]; then
  echo "Now is doing regular caffe training process ..."
  curr_cmds=$exec_cmds
fi

if [ -f ./Makefile.config.serial ]; then
  echo "Now is doing parallelized caffe training process ..."
  curr_cmds=$cpnn_exec_cmds
fi
echo "Current COMMAND configuration: $curr_cmds"

# clean former executable files and object files.
make clean
make

i=0
while read -r line || [ -n "$line" ]; do
   $line

   if [ -f ./Makefile.config.serial ] && [ -f ./LOG/CUPTI-OVERHEAD.csv ]; then
     echo $i
     mv ./LOG/CUPTI-OVERHEAD.csv ./LOG/CUPTI_OVERHEAD_$i.csv
     i=$(($i+1))
   fi
done < $curr_cmds

# PROF || Serial test

if [ -f ./Makefile.config.prof ]; then
  mv ./Makefile.config ./Makefile.config.serial
  mv ./Makefile.config.prof ./Makefile.config
  echo "Now is doing parallel execution ..."
  curr_cmds=$cpnn_exec_cmds
elif [ -f ./Makefile.config.serial ]; then
  mv ./Makefile.config ./Makefile.config.prof
  mv ./Makefile.config.serial ./Makefile.config
  echo "Now is doing serial execution ..."
  curr_cmds=$exec_cmds
fi
echo "Current COMMAND configuration: $curr_cmds"

make clean
make

i=0
while read -r line || [ -n "$line" ]; do
  $line

  echo "Finished $line"
  if [ -f ./Makefile.config.serial ] && [ -f ./LOG/CUPTI-OVERHEAD.csv ]; then
    echo "[CURRENT CUPTI-OVERHEAD]: $i"
    mv ./LOG/CUPTI-OVERHEAD.csv ./LOG/CUPTI-OVERHEAD-$i.csv
    i=$(($i+1))
  fi
done < $curr_cmds
