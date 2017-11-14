#!/usr/bin/env sh
set -e

echo "$@"

./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt $@
