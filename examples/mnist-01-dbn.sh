#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

cd "$basedir/.."

ulimit -t unlimited
time ./bin/deepframe -n1 -m1 -l0.1 -a0.01 -b -i -c examples/mnist-01-dbn.cfg -p \
 -t resources/mnist-01/t10k-images-idx3-ubyte:resources/mnist-01/t10k-labels-idx1-ubyte \
 -s resources/mnist-01/train-images-idx3-ubyte:resources/mnist-01/train-labels-idx1-ubyte $@
