#!/usr/bin/env python
# pystats::test.py

import random
import string
from itertools import izip
from pystats import FreqDist

# array of alphabet chars
letters = string.letters[:14]

fd = FreqDist(is_sample=False)
t = 0

while t < 5000:
	seq = [] ## Generate random strings
	for i in range(1, 3 + random.randint(0,2)):
		seq.append( letters[random.randint(0,13)] )
	fd.add(''.join(seq))	
	t += 1


# Print 100 random samples
for i, (k, v) in izip(xrange(100), fd.iteritems()):
	print "%d %5s: %d" % (i, k, v)

print " "
print "(ABOVE) First 100 Samples"
print " "

print "%7s:  %s" % ('samples', fd.n)
print "%7s:  %s" % ('classes', len(fd.samples))
print "%7s:  %s" % ('sigma', fd.stdev)
print "%7s:  %s" % ('mu', fd.mean)
print "%7s:  %s" % ('median', fd.median)
print "%7s:  %s" % ('range', fd.range)
print "%7s:  %s" % ('min', fd.min)
print "%7s:  %s" % ('max', fd.max)
