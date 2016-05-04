#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

cd "$basedir/.."

ulimit -t unlimited
time ./bin/ffwdnet -m 1000 -n 200 -a0.03 -p \
 -t resources/lines-test.dat -s resources/lines-labels.dat \
 -c examples/lines-dbn.cfg $@
