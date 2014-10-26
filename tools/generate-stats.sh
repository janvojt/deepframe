#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

source "$basedir/vars.sh"

TEST_OUT="$basedir/../$TESTDIR"

for l in $HIDDEN_NEURONS; do
	cat "$TEST_OUT/test-$l.log" \
		| egrep "^[0-9]*[.][0-9]*\\suser" \
		| egrep -o "^[0-9]*[.][0-9]*" \
		> "$TEST_OUT/time-$l.csv"

	cat "$TEST_OUT/test-$l.log" \
		| egrep "^[0-9]*\\smax\\smemory" \
		| egrep -o "^[0-9]*" \
		> "$TEST_OUT/memory-$l.csv"

	cat "$TEST_OUT/test-$l.log" \
		| egrep "\\sMSE\\sof\\s[0-9]*[.][0-9]*" \
		| egrep -o "[0-9]+[.][0-9]+" \
		> "$TEST_OUT/mse-$l.csv"
done
