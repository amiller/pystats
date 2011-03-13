#!/bin/bash

VSN='2.6'
DIST_PACKAGES_PATH="/usr/local/lib/python$VSN/dist-packages"

if [ "$UID" != 0 ]; then
	echo "You must be root."
	exit 1
fi

eval "python setup.py build_ext --inplace" 2>/dev/null
eval "python setup.py clean" 1>/dev/null
eval "rm freqdist.c"

if [ -e "$DIST_PACKAGES_PATH" ]; then
	echo "Putting link in: $DIST_PACKAGES_PATH"
	eval "ln -ns `pwd` $DIST_PACKAGES_PATH"
fi
