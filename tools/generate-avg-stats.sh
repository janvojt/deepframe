#!/bin/bash


script=`readlink -f $0`
basedir=`dirname $script`

source "$basedir/vars.sh"

TEST_OUTPUT="$basedir/../$TESTDIR"

rm -f $TEST_OUTPUT/*-avg*.csv
for t in $MEASURES; do
	rm -f "$TEST_OUTPUT/$t-avg.csv"
	for l in $HIDDEN_NEURONS; do
		total=0
		total2=0
		count=0
		min=99999999999999
		max=0

		for i in $(cat "$TEST_OUTPUT/$t-$l.csv"); do
			total=$(echo $total+$i | bc -l)
			total2=$(echo $total2+$i^2 | bc -l)
			if [ 1 -eq $(echo "$i < $min" | bc -l) ]; then min=$i; fi
			if [ 1 -eq $(echo "$i > $max" | bc -l) ]; then max=$i; fi
			((count++))
		done

		avg=$(echo $total/$count | bc -l)
		avg2=$(echo $total2/$count | bc -l)
		variance=$(echo $avg2-$avg^2 | bc -l)
		varLow=$(echo $avg-$variance | bc -l)
		varHigh=$(echo $avg+$variance | bc -l)

		# check if we are creating graphs for multiple hidden layers
		layers=$(echo "$l" | grep -o "," | wc -l)
		((layers++))
		if [ $layers -gt 1 -o "$l" = "8" ]; then
			echo "$layers,$avg,$min,$varLow,$max,$varHigh" >> "$TEST_OUTPUT/$t-avg-ml.csv"
		else
			echo "$l,$avg,$min,$varLow,$max,$varHigh" >> "$TEST_OUTPUT/$t-avg.csv"
		fi
	done
done
