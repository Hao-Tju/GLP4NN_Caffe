#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver.prototxt -gemmOpt=0 $@

$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver.prototxt -gemmOpt=1 $@

$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver.prototxt -gemmOpt=2 $@

# reduce learning rate by factor of 10 after 8 epochs
# Modifided by Hao Fu.
# $TOOLS/caffe train \
#   --solver=examples/cifar10/cifar10_quick_solver_lr1.prototxt \
#   --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate $@
