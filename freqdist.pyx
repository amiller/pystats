## mupy::lib.pyx

from collections import defaultdict as _defdict

import numpy as _np
cimport numpy as _np

cdef extern from "math.h":
	double sqrt(double)


############################################################
##  STDEV (standard deviation)
############################################################

cdef stdev_i(_np.ndarray array, bool is_sample):
	cdef _np.int_t *data = <_np.int_t*>array.data
	cdef int n = array.shape[0]
	cdef _np.int_t s = 0, s2 = 0
	for i from 0 <= i < n:
		s2 += (data[i] * data[i])
		s += data[i]
	return sqrt((s2 - (s*s)/<double>n) / (n-1 if is_sample else n));

cdef stdev_f(_np.ndarray array, bool is_sample):
	cdef _np.float_t *data = <_np.float_t*>array.data
	cdef int n = array.shape[0]
	cdef _np.float_t s = 0, s2 = 0
	for i from 0 <= i < n:
		s2 += data[i] * data[i]
		s += data[i]
	return sqrt((s2 - (s*s)/n) / (n-1 if is_sample else n));

cdef stdev_d(_np.ndarray array, bool is_sample):
	cdef double *data = <double*>array.data
	cdef int n = array.shape[0]
	cdef double s = 0, s2 = 0
	for i from 0 <= i < n:
		s2 += data[i] * data[i]
		s += data[i]
	return sqrt((s2 - (s*s)/n) / (n-1 if is_sample else n));

cdef stdev_it(it, bool is_sample):
	array = _np.array(it, dtype='double')
	cdef int n = array.shape[0]
	cdef _np.double_t s = 0, s2 = 0
	cdef double t = 0
	for i from 0 <= i < n:
		t = array[i]
		s2 += t*t
		s += t
	return sqrt((s2 - (s*s)/n) / (n-1 if is_sample else n));

############################################################

cpdef _stdev(vector, bool sample=True):
	if isinstance(vector, _np.ndarray):
		if vector.dtype == _np.int:
			return stdev_i(vector, sample)
		elif vector.dtype == _np.float:
			return stdev_f(vector, sample)
		elif vector.dtype == _np.double:
			return stdev_d(vector, sample)
	elif hasattr(vector, '__iter__'):
		return stdev_it(vector, sample)

#cdef class Statistics:
#	stdev = _stdev


cdef class FreqDist(object):

	cdef public bool is_sample

	cdef int _count
	cdef int _median
	cdef int _min
	cdef int _max
	cdef double _sigma
	cdef double _mu
	cdef object _dict
	cdef bool _range_dirty
	cdef bool _median_dirty
	cdef bool _sigma_dirty
	cdef bool _mu_dirty

	def __init__(self, is_sample=True):
		self._dict = _defdict(int)
		self.is_sample = is_sample
		self._min = self._max = 0
		self.reset_flags()
		self._count = 0

	cdef reset_flags(self):
		self._range_dirty = True
		self._median_dirty = True
		self._sigma_dirty = True
		self._mu_dirty = True

	def add(self, sample):
		self.reset_flags()	
		self._count += 1
		self._dict[sample] += 1
		cdef int count = self._dict[sample]
	
	def remove(self, sample):
		self.reset_flags()	
		if self._dict.has_key(sample):
			self._count -= self._dict[sample]
			del self._dict[sample]

	def count(self, sample):
		return self._dict.get(sample, 0)	

	def frequency(self, sample):
		return 1.0 * self._dict[sample] / self._count \
			if self._dict.has_key(sample) else 0
	
	def zscore(self, sample):
		if self._dict.has_key(sample):
			return (self._dict[sample] - self.mean) / self.stdev

	property n:
		def __get__(self):
			return self._count

	property mean:
		def __get__(self):
			if self._mu_dirty and self._count > 0:
				self._mu_dirty = False
				self._mu = self._count / float(len(self._dict))
			return self._mu

	property stdev:
		def __get__(self):
			cdef double s = 0
			if self._sigma_dirty and self._count > 0:
				self._sigma_dirty = False
				self._sigma = stdev_i(
					_np.array(self._dict.values(), dtype=int), 
					self.is_sample)
			return self._sigma

	property variance:
		def __get__(self):
			cdef double s = self.stdev
			return s * s

	property median:
		def __get__(self):
			cdef int n
			if self._count == 0:
				return 0
			if self._median_dirty:
				self._median_dirty = False
				freqs = sorted(self._dict.values())
				n = len(freqs)
				if (n % 2 == 0):
					self._median = freqs[n/2]
				else:
					self._median = (freqs[n/2] + freqs[n/2+1]) / 2.0
			return self._median

	property min:
		def __get__(self):
			if self._range_dirty:
				self._range_dirty = False
				a = _np.array(self._dict.values())
				self._max = a.max()
				self._min = a.min()
			return self._min

	property max:
		def __get__(self):
			if self._range_dirty:
				self._range_dirty = False
				a = _np.array(self._dict.values())
				self._max = a.max()
				self._min = a.min()
			return self._max
	
	property range:
		def __get__(self):
			if self._range_dirty:
				self._range_dirty = False
				a = _np.array(self._dict.values())
				self._max = a.max()
				self._min = a.min()
			return self._max - self._min

	property samples:
		def __get__(self):
			return self._dict.keys()

	def iteritems(self):
		return self._dict.iteritems()

	def __getitem__(self, sample):
		return self._dict[sample]

	def __contains__(self, sample):
		return (sample in self._dict)

	def __repr__(self):
		return "<FreqDist(N=%d)>" % (self._count)


