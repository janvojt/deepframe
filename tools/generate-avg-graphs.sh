#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`"/.."

source "$1"

TEST_OUT="$basedir/$TESTDIR"
STATS_OUT="$TESTDIR/stats"
GRAPHS_OUT="$TEST_OUT/graphs"

# Compute boxwidth for gnuplot
if [ -z "$CONF_DIR" ]; then
	ITER_CONF="$HIDDEN_NEURONS"
	# find number of neurons in a layer for multilayer testing
	ML_NEURONS=$(echo $HIDDEN_NEURONS | tr " " "\n" | grep "," | tail -n1 | grep -o "[0-9]*" | head -n1)
else
	ITER_CONF=`ls "$basedir/$CONF_DIR/"*".cfg"`
	# find number of neurons in a layer for multilayer testing
	ML_NEURONS=$(echo $ITER_CONF | tr " " "\n" | sed 's/^.*\///' | sed 's/\.[^.]*$//' | grep "," | tail -n1 | grep -o "[0-9]*" | head -n1)
fi

min_sl=99999999
max_sl=-99999999
min_ml=99999999
max_ml=-99999999
for l in $ITER_CONF; do

	# Prepare X-axis values
	if [ -z "$CONF_DIR" ]; then
		xaxis="$l"
	else
		xaxis=`echo "$l" | sed 's/^.*\///' | sed 's/\.[^.]*$//'`
	fi

	# if this graph is for multiple layers...
	layers=$(echo "$xaxis" | grep -o "," | wc -l)
	((layers++))

	if [ "$xaxis" = "$ML_NEURONS" ]; then
		min_ml=$(($min_ml>1 ? 1 : $min_ml))
		max_ml=$(($max_ml<1 ? 1 : $max_ml))
	fi

	if [ $layers -gt 1 ]; then
		if [ 1 -eq $(echo "$min_ml > $layers" | bc -l) ]; then min_ml=$layers; fi
		if [ 1 -eq $(echo "$max_ml < $layers" | bc -l) ]; then max_ml=$layers; fi
	else
		if [ 1 -eq $(echo "$min_sl > $xaxis" | bc -l) ]; then min_sl=$xaxis; fi
		if [ 1 -eq $(echo "$max_sl < $xaxis" | bc -l) ]; then max_sl=$xaxis; fi
	fi
done

BOX_WIDTH_SL=$(echo "0.04 * ($max_sl - $min_sl)" | bc -l)
BOX_WIDTH_ML=$(echo "0.04 * ($max_ml - $min_ml)" | bc -l)
# END of computing boxwidth

# Generate graphs
mkdir -p "$GRAPHS_OUT"
for t in $MEASURES; do

	if [ -f "$basedir/$STATS_OUT/$t-avg.csv" ]; then
		"$basedir/tools/graphs/generate-avg-$t.sh" $STATS_OUT "$PROBLEM_TITLE" \
		"$BOX_WIDTH_SL" > "$GRAPHS_OUT/graph-$t-avg.eps"
	fi

	if [ -f "$basedir/$STATS_OUT/$t-avg-ml.csv" ]; then
		"$basedir/tools/graphs/generate-avg-$t.sh" $STATS_OUT "$PROBLEM_TITLE" \
		"$BOX_WIDTH_ML" -ml > "$GRAPHS_OUT/graph-$t-avg-ml.eps"
	fi
done
