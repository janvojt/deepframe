<?php

function dec2bin($dec, $pad) {
	$formatted = '';
	$bin = str_pad(decbin($dec), $pad, '0', STR_PAD_LEFT);
	return implode(' ', str_split($bin));
}

echo "64\n";

for ($a = 0; $a < 8; $a++) {
	for ($b = 0; $b < 8; $b++) {
		echo dec2bin($a, 3);
		echo ' ';
		echo dec2bin($b, 3);
		echo ' > ';
		echo dec2bin($a + $b, 4);
		echo "\n";
	}
}

