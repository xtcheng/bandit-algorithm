import numpy as np


def unit(vector):
	temp_len = 0
	for a in vector:
		temp_len += float(a**2) # cast to float because it may be a sympy-object. And numpy doesn't like that.
	return (1/np.sqrt(temp_len)) * vector


def vectorLen(vector):
	temp_len = 0
	for a in vector:
		temp_len += float(a**2)
	return temp_len
