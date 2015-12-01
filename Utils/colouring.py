# coding: utf-8
#!/usr/bin/env python

'''
    Filename: colouring.py
    Author: Xiaodong Qin
    Email: michael.chin@sydney.edu.au
    
    This function applies colours on data.
    
    Parameters:
        data:      a 2-d numpy array 
        colours:   a list of colours to be used to create colour map
        steps:     a list of values in data to be used to intepolate colours
                   the 'steps' list must have the same size as 'colours' list
    Return:
        coloured data
		
	Example:
	colours = [
			[0,0,255],
			[0,255,255],
			[0,255,0],
			[0,255,0],
			[255,255,0],
			[255,0,0],
		]
	steps = [-200, -10, -1, 1, 10, 200]
		
	rgb = colouring(np.array([[1,1],[2,3]]),colours, steps)
'''
import shading
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def colouring(data, colours, steps):
    colour_list = []
    for i in range(len(steps)):
        colour_list.append((float(steps[i]-steps[0])/(steps[-1]-steps[0]), [x/255.0 for x in colours[i]]))

    my_cmap = LinearSegmentedColormap.from_list('my_cmap', colour_list, N=1024)
    return shading.shade(data, cmap=my_cmap, intensity=shading.intensity(data))
