#!/bin/bash

time ./bin/ffwdnet -r 2 -m 10000 -t resources/4bitsum-test.dat -s resources/4bitsum-labels.dat -l 8,53,53,5 $@
