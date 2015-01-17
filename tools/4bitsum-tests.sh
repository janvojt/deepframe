#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

source "$basedir/vars.sh"

TEST_OUT="$basedir/../$TESTDIR"
DATASET_LABELS="$basedir/../resources/4bitsum-labels.dat"
DATASET_TESTS="$basedir/../resources/4bitsum-test.dat"
EXEC="$basedir/../bin/ffwdnet"

#rm -rf "$TEST_OUT"
mkdir -p "$TEST_OUT"

for l in $HIDDEN_NEURONS; do
for (( i=1;i<=$ITERATIONS;i++ )); do
	(/usr/bin/time -f "%U user\n%S system\n%e real\n%M max memory (kB)\n" \
		"$EXEC" -m $EPOCHS \
		-e -1 \
		-s "$DATASET_LABELS" \
		-t "$DATASET_TESTS" \
		-c "8,$l,5" \
		) &>> "$TEST_OUT/test-$l.log"
done
done
