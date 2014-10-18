<?php

function dec2bin($dec, $pad) {
	$bin = str_pad(decbin($dec), $pad, '0', STR_PAD_LEFT);
	return implode(' ', str_split($bin));
}

$n = array_key_exists(1, $argv) ? $argv[1] : 2;
$max = pow(2, $n);
$lines = $max * $max;

echo $lines . "\n";

for ($a = 0; $a < $max; $a++) {
	for ($b = 0; $b < $max; $b++) {
		echo dec2bin($a, $n);
		echo ' ';
		echo dec2bin($b, $n);
		echo ' > ';
		echo dec2bin($a + $b, $n + 1);
		echo "\n";
	}
}

