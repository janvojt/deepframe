#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

cd "$basedir/.."

ulimit -t unlimited
time ./bin/deepframe -m 10000 -l 0.1 -a-0.3,0.3 \
 -t resources/lines-test.dat -s resources/lines-labels.dat \
 -c examples/lines-conv.cfg $@
