#!/bin/bash
#
# THIS SCRIPT SHOULD NOT BE CALLED DIRECTLY!
# Call through ../generate-avg-graphs.sh instead.
#

script=`readlink -f $0`
basedir=`dirname $script`

TESTDIR="$1"
FILE_SUFFIX="$2"
TEST_OUT="$basedir/../../$TESTDIR"

gnuplot << GNUEOS
reset
set terminal eps

set datafile separator ","

set xlabel "hidden neurons"

set ylabel "Max memory (kB)"

set title "Memory requirements for 4-bit sum with 1 hidden layer"
set key reverse Left outside
set grid

set style data points

# compute linear fit
set fit quiet
set fit logfile '/dev/null'
f(x) = m * x + b
fit f(x) "$TEST_OUT/memory-avg$FILE_SUFFIX.csv" using 1:2 via m,b

plot "$TEST_OUT/memory-avg$FILE_SUFFIX.csv" using 1:2 title "Measured" \
     ,f(x) title "Linear fit"

GNUEOS
