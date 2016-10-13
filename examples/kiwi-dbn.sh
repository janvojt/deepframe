#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

cd "$basedir/.."

ulimit -t unlimited
time ./bin/deepframe -n1 -m100 -e0.01 -l0.1 -a0.01 -g -c examples/kiwi-dbn.cfg -p \
 -t ~/workspace/kiwi/big.dat \
 -s ~/workspace/kiwi/small.dat $@
