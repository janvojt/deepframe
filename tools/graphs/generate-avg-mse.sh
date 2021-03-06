#!/bin/bash
#
# THIS SCRIPT SHOULD NOT BE CALLED DIRECTLY!
# Call through ../generate-avg-graphs.sh instead.
#

script=`readlink -f $0`
basedir=`dirname $script`

TESTDIR="$1"
PROBLEM_TITLE="$2"
BOXWIDTH="$3"
FILE_SUFFIX="$4"
TEST_OUT="$basedir/../../$TESTDIR"

if [ -z "$FILE_SUFFIX" ]; then
	TITLE_MAIN="MSE for $PROBLEM_TITLE with 1 hidden layer."
	TITLE_X="hidden neurons"
else
	TITLE_MAIN="MSE for $PROBLEM_TITLE."
	TITLE_X="hidden layers"
fi

gnuplot << GNUEOS
reset
set terminal eps

set datafile separator ","

set xlabel "$TITLE_X"

set ylabel "Mean Square Error"

set title "$TITLE_MAIN"
set key reverse Left outside
set grid

set style data points

set boxwidth $BOXWIDTH

# compute linear fit
set fit quiet
set fit logfile '/dev/null'
f(x) = m * x + b
fit f(x) "$TEST_OUT/mse-avg$FILE_SUFFIX.csv" using 1:2 via m,b

plot "$TEST_OUT/mse-avg$FILE_SUFFIX.csv" \
	using 1:2:3:5:2 with candlesticks title "Measured" whiskerbars \
	, '' using 1:2:2:2:2 with candlesticks fs solid title "Mean" \
	,f(x) title "Linear fit"

GNUEOS
