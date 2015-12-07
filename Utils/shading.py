#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
	The shading.py applies shading to raster data.
	Copyright (C) 2015 The University of Sydney, Australia

	This program is free software; you can redistribute it and/or
	modify it under the terms of the GNU General Public License, version 2,
	as published by the Free Software Foundation.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program; if not, write to the Free Software
	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

	*******************************************************************************
	
    Filename: shading.py
    Author: Xiaodong Qin
    Email: michael.chin@sydney.edu.au
    
	Get latest version and submit bug report at 
	GitHub: https://github.com/GPlates/Portal/blob/master/Utils/shading.py
'''	
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt

'''
    This function applies intensity on grid data.
    The algorithm is based on PEGTOP soft light mode.
    http://www.pegtop.net/delphi/articles/blendmodes/softlight.htm
    Parameters:
        im:        a 2-d numpy array 
        intensity: a 2-d numpy array of the same size as im
        cmap:      a matplotlib colormap 
        vmax:      the max value that will be used to interpolate cmap 
        vmin:      the min value that will be used to interpolate cmap 
    Return:
        the coloured and shaded RBG data
'''
def shade(im, intensity, cmap=cm.jet, vmax=None, vmin=None ):
    v_max = vmax
    v_min = vmin
    if not vmax:
        v_max = im.max()
    if not vmin:
        v_min = im.min()
    
    a = cmap((im-v_min)/float(v_max-v_min))[:,:,:3]
    
    intensity = (intensity - intensity.min())/(intensity.max() - intensity.min())
    b = intensity.repeat(3).reshape(rgb.shape)
    a = 2*a*b + (a**2)*(1-2*b)
    return a

''' 
    This function calculate the intensity of data.
    This program is based on Hillshade algorithm
    http://edndoc.esri.com/arcobjects/9.2/net/shared/geoprocessing/spatial_analyst_tools/how_hillshade_works.htm
    Parameters: 
        data:     a 2-d numpy array 
        azimuth:  the direction of light source (0-360 degrees)
        altitude: the altitude angle of light source (0-90 degrees)
		z_factor: factor to control the terrain height 
    Return: 
		a 2-d array of normalized intensity 
'''
def intensity(data, azimuth =165.0, altitude=45.0, z_factor=0.1):
    azimuth_radian = (90-azimuth)*np.pi/180.0
    altitude_radian = (90-altitude)*np.pi/180.0

    dx, dy = np.gradient(data)
    
    slope_radian = np.arctan(z_factor*np.sqrt(dx**2 + dy**2))
    aspect_radian = np.arctan2(-dx, dy)
    
    intensity =np.cos(altitude_radian )*np.cos(slope_radian) + \
		np.sin(altitude_radian )*np.sin(slope_radian)*np.cos(azimuth_radian - aspect_radian )
    
    intensity = (intensity - intensity.min())/(intensity.max() - intensity.min())
    return intensity

