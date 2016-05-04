#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

cd "$basedir/.."

ulimit -t unlimited
time ./bin/ffwdnet -n1 -m1 -l0.1 -a0.01 -b -i -c examples/mnist-dbn.cfg -p \
 -t resources/mnist/t10k-images-idx3-ubyte:resources/mnist/t10k-labels-idx1-ubyte \
 -s resources/mnist/train-images-idx3-ubyte:resources/mnist/train-labels-idx1-ubyte $@
