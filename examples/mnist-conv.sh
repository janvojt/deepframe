#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

cd "$basedir/.."

ulimit -t unlimited
time ./bin/ffwdnet -l 0.001 -a-0.3,0.3 -b -i -c examples/mnist-conv.cfg \
 -t resources/mnist/t10k-images-idx3-ubyte:resources/mnist/t10k-labels-idx1-ubyte \
 -s resources/mnist/train-images-idx3-ubyte:resources/mnist/train-labels-idx1-ubyte $@
