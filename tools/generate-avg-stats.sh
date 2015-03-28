#!/bin/bash


script=`readlink -f $0`
basedir=`dirname $script`

source "$1"

TEST_OUT="$basedir/../$TESTDIR"
STATS_OUT="$TEST_OUT/stats"


mkdir -p "$STATS_OUT"
rm -f $STATS_OUT/*-avg*.csv

# find number of neurons in a layer for multilayer testing
ML_NEURONS=$(echo $HIDDEN_NEURONS | tr " " "\n" | grep "," | tail -n1 | grep -o "[0-9]*" | head -n1)

for t in $MEASURES; do

	echo "Computing stats for $t."

	rm -f "$STATS_OUT/$t-avg.csv"
	for l in $HIDDEN_NEURONS; do

		echo "Computing layer configuration $l."

		total=0
		total2=0
		count=0
		min=99999999999999
		max=0
		for i in $(cat "$STATS_OUT/$t-$l.csv"); do
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

		if [ "$l" = "$ML_NEURONS" ]; then
                        echo "$layers,$avg,$min,$varLow,$max,$varHigh" >> "$STATS_OUT/$t-avg-ml.csv"
		fi

		if [ $layers -gt 1 ]; then
			echo "$layers,$avg,$min,$varLow,$max,$varHigh" >> "$STATS_OUT/$t-avg-ml.csv"
		else
			echo "$l,$avg,$min,$varLow,$max,$varHigh" >> "$STATS_OUT/$t-avg.csv"
		fi
	done
done
