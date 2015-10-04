#!/bin/bash

ulimit -t unlimited
time ./bin/ffwdnet -m 10000 -l 0.1 -a-0.3,0.3 \
 -t resources/lines-test.dat -s resources/lines-labels.dat \
 -c examples/lines-conv.cfg $@
