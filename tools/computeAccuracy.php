#!/usr/bin/php
<?php

if (!array_key_exists(1, $argv)) {
	echo "Please provide path to the file with network output as the first argument.\n";
	exit(1);
}

$out = $argv[1];

if (!file_exists($out)) {
	echo "Given file path '$out' is not accessible.\n";
	exit(2);
}


$fp = fopen($out, 'rb');

$regex = '/^Output\sfor\spattern\s([0-9]+).*\[\s([^\]]*)\s\].*\[\s([^\]]*)\s\]/';

$correct = 0;
$total = 0;
$firstRun = true;
if ($fp) {
	while (($line = fgets($fp)) !== false) {
		if (preg_match($regex, $line, $matches)) {
			$pattern = $matches[1];
			$output = explode(", ", $matches[2]);
			$labels = explode(", ", $matches[3]);
			$outputMax = array_keys($output, max($output));
			$labelsMax = array_keys($labels, max($labels));

			$total++;
			$correct += $outputMax[0] == $labelsMax[0] ? 1 : 0;
		} else if (preg_match("/Compiled with [0-9.]+bit precision/", $line, $matches)) {

			// if this is first run,
			// we still need to collect results
			if ($firstRun) {
				$firstRun = false;
				continue;
			}

			// reached end of a single run
			// -> print result and continue with next runs (if any)
			echo ($correct / $total) . "\n";
			$total = 0;
			$correct = 0;
		}
	}
	fclose($fp);

	if ($total > 0) {
		// we have not printed result for some patterns
		echo ($correct / $total) . "\n";
	}

} else {
	echo "File '$out' cannot be open for reading.\n";
	exit(3);
}

