#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

cd "$basedir/.."

ulimit -t unlimited
time ./bin/deepframe -l 0.001 -a-0.3,0.3 -e 0.04 -i -c examples/mnist-conv.cfg \
 -t resources/mnist/t10k-images-idx3-ubyte:resources/mnist/t10k-labels-idx1-ubyte \
 -s resources/mnist/train-images-idx3-ubyte:resources/mnist/train-labels-idx1-ubyte $@
