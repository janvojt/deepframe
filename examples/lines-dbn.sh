#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

cd "$basedir/.."

ulimit -t unlimited
time ./bin/ffwdnet -m 10000 -n 100 -l 1 -a-0.3,0.3 \
 -t resources/lines-test.dat -s resources/lines-labels.dat \
 -c examples/lines-dbn.cfg $@
