#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=models/bvlc_reference_caffenet/solver.prototxt -gemmOpt=0 $@
#./build/tools/caffe train \
#    --solver=models/bvlc_reference_caffenet/solver.prototxt -gemmOpt=1 $@
#./build/tools/caffe train \
#    --solver=models/bvlc_reference_caffenet/solver.prototxt -gemmOpt=2 $@
