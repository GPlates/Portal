
# coding: utf-8

# In[ ]:

#!/usr/bin/env python


from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt

'''
    Filename: shading.py
    Author: Xiaodong Qin
    Email: michael.chin@sydney.edu.au
    
    This function applies intensity on data.
    The algorithm is based on PEGTOP soft light mode.
    http://www.pegtop.net/delphi/articles/blendmodes/softlight.htm
    Parameters:
        im:        a 2-d array 
        intensity: a 2-d array of same size as im
                    
        cmap:      a matplotlib colormap 
        vmax:      the max value for this color map
        vmin:      the min value for this color map
    Return:
        data which has been coloured and shaded.
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
    This function calculate the intensity of given data.
    This program is based on Hillshade algorith 
    http://edndoc.esri.com/arcobjects/9.2/net/shared/geoprocessing/spatial_analyst_tools/how_hillshade_works.htm
    Parameters: 
         data: a 2-d array of data
         azimuth: the direction of light source
         altitude: the altitude angle of light source
    Return: a 2-d array of normalized intensity 
'''
def intensity(data, azimuth =165.0, altitude=45.0):
    azimuth_radian = (90-azimuth)*np.pi/180.0
    altitude_radian = (90-altitude)*np.pi/180.0

    dx, dy = np.gradient(data)
    
    slope_radian = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_radian = np.arctan2(-dx, -dy)
    
    intensity =np.cos(altitude_radian )*np.cos(slope_radian) +         np.sin(altitude_radian )*np.sin(slope_radian)*np.cos(azimuth_radian - aspect_radian )
    
    intensity = (intensity - intensity.min())/(intensity.max() - intensity.min())
    return intensity

