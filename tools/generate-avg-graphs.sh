#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

source "$basedir/vars.sh"

TEST_OUT="$basedir/../$TESTDIR"


for t in $MEASURES; do

	"$basedir/graphs/generate-avg-$t.sh" $TESTDIR \
		> "$TEST_OUT/graph-$t-avg.eps"

	for l in $HIDDEN_NEURONS; do
		# check if we are creating graphs for multiple hidden layers
		layers=$(echo "$l" | grep -o "," | wc -l)
		((layers++))
		if [ $layers -gt 1 -o "$l" = "8" ]; then
			"$basedir/graphs/generate-avg-$t.sh" $TESTDIR -ml \
				> "$TEST_OUT/graph-$t-avg-ml.eps"
		fi
	done
done
