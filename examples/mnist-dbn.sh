#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

cd "$basedir/.."

ulimit -t unlimited
time ./bin/ffwdnet -n 100 -l 0.1 -a-0.3,0.3 -b -i -c examples/mnist-dbn.cfg -p \
 -t resources/mnist/t10k-images-idx3-ubyte:resources/mnist/t10k-labels-idx1-ubyte \
 -s resources/mnist/train-images-idx3-ubyte:resources/mnist/train-labels-idx1-ubyte $@
