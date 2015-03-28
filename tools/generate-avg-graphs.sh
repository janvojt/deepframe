#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

source "$1"

TEST_OUT="$basedir/../$TESTDIR"
STATS_OUT="$TESTDIR/stats"
GRAPHS_OUT="$TEST_OUT/graphs"

mkdir -p "$GRAPHS_OUT"
for t in $MEASURES; do

	"$basedir/graphs/generate-avg-$t.sh" $STATS_OUT "$PROBLEM_TITLE" \
		> "$GRAPHS_OUT/graph-$t-avg.eps"

	l=$(echo $HIDDEN_NEURONS | tr " " "\n" | grep "," | tail -n1)
	layers=$(echo "$l" | grep -o "," | wc -l)
	((layers++))
	if [ $layers -gt 1 -o "$l" = "8" ]; then
		"$basedir/graphs/generate-avg-$t.sh" $STATS_OUT "$PROBLEM_TITLE" -ml \
			> "$GRAPHS_OUT/graph-$t-avg-ml.eps"
	fi
done
