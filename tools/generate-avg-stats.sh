#!/bin/bash


script=`readlink -f $0`
basedir=`dirname $script`

source "$basedir/vars.sh"

TEST_OUTPUT="$basedir/../$TESTDIR"

rm -f $TEST_OUTPUT/*-avg-ml.csv
for t in $MEASURES; do
	rm -f "$TEST_OUTPUT/$t-avg.csv"
	for l in $HIDDEN_NEURONS; do
		total=0
		count=0
		for i in $(cat "$TEST_OUTPUT/$t-$l.csv"); do
			total=$(echo $total+$i | bc -l)
			((count++))
		done
		avg=$(echo $total/$count | bc -l)
		
		# check if we are creating graphs for multiple hidden layers
		layers=$(echo "$l" | grep -o "," | wc -l)
		((layers++))
		if [ $layers -gt 1 -o "$l" = "8" ]; then
			echo "$layers,$avg" >> "$TEST_OUTPUT/$t-avg-ml.csv"
		else
			echo "$l,$avg" >> "$TEST_OUTPUT/$t-avg.csv"
		fi
	done
done
