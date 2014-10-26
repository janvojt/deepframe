#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

source "$basedir/vars.sh"

TEST_OUT="$basedir/../$TESTDIR"


for l in $HIDDEN_NEURONS; do
	for t in $MEASURES; do
		"$basedir/graphs/generate-$t.sh" $TESTDIR $l \
			> "$TEST_OUT/graph-$t-$l.eps"
	done
done
