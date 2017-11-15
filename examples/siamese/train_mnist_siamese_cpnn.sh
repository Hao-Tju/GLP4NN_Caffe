#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/siamese/mnist_siamese_solver.prototxt -gemmOpt=0 $@
$TOOLS/caffe train --solver=examples/siamese/mnist_siamese_solver.prototxt -gemmOpt=1 $@
$TOOLS/caffe train --solver=examples/siamese/mnist_siamese_solver.prototxt -gemmOpt=2 $@
