#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`"/.."

source "$1"

TEST_OUT="$basedir/$TESTDIR"
STATS_OUT="$TESTDIR/stats"
GRAPHS_OUT="$TEST_OUT/graphs"

# Prepare configuration
if [ -z "$CONF_DIR" ]; then
        ITER_CONF="$HIDDEN_NEURONS"
else
        ITER_CONF=`ls "$basedir/$CONF_DIR/"*".cfg"`
fi

mkdir -p "$GRAPHS_OUT"
for l in $ITER_CONF; do

	# Prepare X-axis values
        if [ -z "$CONF_DIR" ]; then
                xaxis="$l"
        else
                xaxis=`echo "$l" | sed 's/^.*\///' | sed 's/\.[^.]*$//'`
        fi

	for t in $MEASURES; do
		"$basedir/tools/graphs/generate-$t.sh" $STATS_OUT $xaxis "$PROBLEM_TITLE" \
			> "$GRAPHS_OUT/graph-$t-$xaxis.eps"
	done
done
