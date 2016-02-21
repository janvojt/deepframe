#!/bin/bash
#
# This script launches a set of tests testing different network architectures
# and learner configurations.


# Disable limits on process CPU time if any
ulimit -t unlimited

script=`readlink -f $0`
basedir=`dirname $script`"/.."

TMPDIR="/tmp"

# Process options
USAGE="Usage: $0 configuration-file [network-overrides]"
if [ "$#" == "0" ]; then
	echo "$USAGE"
	exit 1
elif [ ! -f "$1" ]; then
	echo "File '$1' not found."
	echo "$USAGE"
	exit 1
fi

# Read configuration
source "$1"
shift

if [ -n "$CONF_DIR" -a "(" ! -d "$basedir/$CONF_DIR" ")" ]; then
	echo "Directory with network configurations could not be found at '$basedir/$CONF_DIR'."
	exit 1
fi

# Prepare configuration
if [ -z "$CONF_DIR" ]; then
	ITER_CONF="$HIDDEN_NEURONS"
else
	ITER_CONF=`ls "$basedir/$CONF_DIR/"*".cfg"`
fi

# Run the tests
TEST_OUT="$TESTDIR"
RESOURCES_DIR="resources"
EXEC="bin/ffwdnet"

cd "$basedir"
mkdir -p "$TEST_OUT"

for l in $ITER_CONF; do
	echo "Running configuration $l ..."
for (( i=1;i<=$ITERATIONS;i++ )); do

	# Prepare configuration
	if [ -z "$CONF_DIR" ]; then
		conf="$INPUT_NEURONS,$l,$OUTPUT_NEURONS"
		xaxis="$l"
	else
		conf="$l"
		xaxis=`echo "$l" | sed 's/^.*\///' | sed 's/\.[^.]*$//'`
	fi

	# Check for NaNs in results, and retry the run if we come across them.
	FOUNDNAN="init for first run"
	while [ "$FOUNDNAN" ]; do

		echo "Running iteration $i ..."

		FOUNDNAN=`/usr/bin/time -f "%U user\n%S system\n%e real\n%M max memory (kB)\n" \
			"$EXEC" \
			-m "$EPOCHS" \
			-n 100 \
			-l "$LEARNING_RATE" \
			-a "$INIT_INTERVALS" \
			-e -1 \
			-s "$DATASET_LABELS" \
			-t "$DATASET_TESTS" \
			-c "$conf" \
			"$GPU_FLAG" \
			$ADD_OPTS \
			$@ \
			2>&1 | tee "$TMPDIR/test-$xaxis.log" \
			| sed -n '/NaN/{p;q;}' \
			| tee -a "$TEST_OUT/test-$xaxis.log"`

		# save output only if there are no NaNs
		test ! "$FOUNDNAN" && cat "$TMPDIR/test-$xaxis.log" >> "$TEST_OUT/test-$xaxis.log"

		# Clean the temp file with the output of last run
		rm "$TMPDIR/test-$xaxis.log"
	done
done
done
