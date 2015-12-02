#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Filename: histeq.py
    Author: Xiaodong Qin
    Email: michael.chin@sydney.edu.au

	Get latest version and submit bug report at 
	GitHub: https://github.com/GPlates/Portal/blob/master/Utils/histeq.py
	
    This function returns normalized histogram equalized data.
    
    Parameters:
       im:       two-dimensional numpy array
       num_bins: the number of bins
       vmin:     the minimun value that will be used in calculating histogram
       vmax:     the maximun value that will be used in calculating histogram
    Returns:
       new_im: the histogram equalized and normalized data
       cdf : cumulative distribution function
	   
	References:
    https://en.wikipedia.org/wiki/Histogram_equalization
'''

import numpy

def hiseq(im, num_bins=1024, vmin=numpy.nan, vmax=numpy.nan):
    if numpy.isnan(vmin):
        min_value=numpy.nanmin(im)
    else:
        min_value=vmin
    if numpy.isnan(vmax):
        max_value=numpy.nanmax(im)
    else:
        max_value=vmax
    
    where_are_NaNs = numpy.isnan(im) 
    im[where_are_NaNs] = min_value-1 

    #histogram
    #imhist contains the number of values in each bin
    #bins contains the boundaries of bins
    imhist,bins = numpy.histogram(
        im.flatten(), 
        num_bins, 
        range=(min_value, max_value),
        normed=False)
    
    #the more values in a bin, the bigger range into which the values will be transformed. 
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = cdf / float(cdf[-1]) #normalize
    #use linear interpolation to transform data
    new_im = numpy.interp(im.flatten(),bins[:-1],cdf).reshape(im.shape)
    
    new_im[where_are_NaNs] = numpy.nan
    im[where_are_NaNs] = numpy.nan
    
    return new_im, cdf

