#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

cd "$basedir/.."

time ./bin/ffwdnet -i -t resources/mnist/t10k-images-idx3-ubyte:resources/mnist/t10k-labels-idx1-ubyte -s resources/mnist/train-images-idx3-ubyte:resources/mnist/train-labels-idx1-ubyte -c 784,784,10 -m 200 $@
