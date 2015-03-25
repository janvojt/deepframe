#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

source "$1"

TEST_OUT="$basedir/../$TESTDIR"
STATS_OUT="$TESTDIR/stats"
GRAPHS_OUT="$TEST_OUT/graphs"

mkdir -p "$GRAPHS_OUT"
for l in $HIDDEN_NEURONS; do
	for t in $MEASURES; do
		"$basedir/graphs/generate-$t.sh" $STATS_OUT $l "$PROBLEM_TITLE" \
			> "$GRAPHS_OUT/graph-$t-$l.eps"
	done
done
