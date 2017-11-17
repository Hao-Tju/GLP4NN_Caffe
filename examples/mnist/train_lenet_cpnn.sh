#!/usr/bin/env sh
set -e

echo "$@"

./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt -gemmOpt=0 $@
./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt -gemmOpt=1 $@
./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt -gemmOpt=2 $@
