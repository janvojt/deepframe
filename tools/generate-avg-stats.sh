#!/bin/bash


script=`readlink -f $0`
basedir=`dirname $script`"/.."

source "$1"

TEST_OUT="$basedir/$TESTDIR"
STATS_OUT="$TEST_OUT/stats"


mkdir -p "$STATS_OUT"
rm -f $STATS_OUT/*-avg*.csv

# Prepare configuration
if [ -z "$CONF_DIR" ]; then
        ITER_CONF="$HIDDEN_NEURONS"
	# find number of neurons in a layer for multilayer testing
	ML_NEURONS=$(echo $HIDDEN_NEURONS | tr " " "\n" | grep "," | tail -n1 | grep -o "[0-9]*" | head -n1)
else
        ITER_CONF=`ls "$basedir/$CONF_DIR/"*".cfg"`
	# find number of neurons in a layer for multilayer testing
	ML_NEURONS=$(echo $ITER_CONF | tr " " "\n" | sed 's/^.*\///' | sed 's/\.[^.]*$//' | grep "," | tail -n1 | grep -o "[0-9]*" | head -n1)
fi


for t in $MEASURES; do

	echo "Computing stats for $t."

	rm -f "$STATS_OUT/$t-avg.csv"
	for l in $ITER_CONF; do

		echo "Computing layer configuration $l."

		# Prepare X-axis values
        	if [ -z "$CONF_DIR" ]; then
	                xaxis="$l"
        	else
                	xaxis=`echo "$l" | sed 's/^.*\///' | sed 's/\.[^.]*$//'`
	        fi

		total=0
		total2=0
		count=0
		min=99999999999999
		max=0
		for i in $(cat "$STATS_OUT/$t-$xaxis.csv"); do
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
		layers=$(echo "$xaxis" | grep -o "," | wc -l)
		((layers++))

		if [ "$xaxis" = "$ML_NEURONS" ]; then
                        echo "$layers,$avg,$min,$varLow,$max,$varHigh" >> "$STATS_OUT/$t-avg-ml.csv"
		fi

		if [ $layers -gt 1 ]; then
			echo "$layers,$avg,$min,$varLow,$max,$varHigh" >> "$STATS_OUT/$t-avg-ml.csv"
		else
			echo "$xaxis,$avg,$min,$varLow,$max,$varHigh" >> "$STATS_OUT/$t-avg.csv"
		fi
	done
done
