#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`"/.."

source "$1"

TEST_OUT="$basedir/$TESTDIR"
STATS_OUT="$TEST_OUT/stats"

# Prepare configuration
if [ -z "$CONF_DIR" ]; then
        ITER_CONF="$HIDDEN_NEURONS"
else
        ITER_CONF=`ls "$basedir/$CONF_DIR/"*".cfg"`
fi

mkdir -p "$STATS_OUT"
for l in $ITER_CONF; do

	# Prepare X-axis values
        if [ -z "$CONF_DIR" ]; then
                xaxis="$l"
        else
                xaxis=`echo "$l" | sed 's/^.*\///' | sed 's/\.[^.]*$//'`
        fi


	cat "$TEST_OUT/test-$xaxis.log" \
		| egrep "^[0-9]*[.][0-9]*\\suser" \
		| egrep -o "^[0-9]*[.][0-9]*" \
		> "$STATS_OUT/time-$xaxis.csv"

	cat "$TEST_OUT/test-$xaxis.log" \
		| egrep "^[0-9]*\\smax\\smemory" \
		| egrep -o "^[0-9]*" \
		> "$STATS_OUT/memory-$xaxis.csv"

	cat "$TEST_OUT/test-$xaxis.log" \
		| egrep "Training\\s.*\\sMSE\\sof\\s[0-9]*[.][0-9]*" \
		| egrep -o "[0-9]+[.][0-9]+" \
		> "$STATS_OUT/mse-$xaxis.csv"

        php "$basedir/tools/computeMnistSuccess.php" "$TEST_OUT/test-$xaxis.log" \
                > "$STATS_OUT/acc-$xaxis.csv"

done
