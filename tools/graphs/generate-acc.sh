#!/bin/bash
#
# THIS SCRIPT SHOULD NOT BE CALLED DIRECTLY!
# Call through ../generate-graphs.sh instead.
#

script=`readlink -f $0`
basedir=`dirname $script`

TESTDIR="$1"
LAYERS=$2
PROBLEM_TITLE="$3"
TEST_OUT="$basedir/../../$TESTDIR"

gnuplot << GNUEOS
reset
set terminal eps

set xlabel "trial"

set ylabel "Accuracy"

# show zero on y-axis
set yrange [0:*]

set title "Accuracy for $PROBLEM_TITLE with $LAYERS hidden neurons"
set key reverse Left outside
set grid

set style data points

# compute interesting stats about our data
#stats "$TEST_OUT/acc-$LAYERS.csv" index 0 using 1 prefix "A"

plot "$TEST_OUT/acc-$LAYERS.csv" using 0:1 title "Measured"
GNUEOS
