#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

cd "$basedir/.."

ulimit -t unlimited
time ./bin/deepframe -n1 -m2 -e0.001 -g -c examples/kiwi-dbn.cfg \
 -t ../kiwi/small.dat \
 -s ../kiwi/big.dat $@
