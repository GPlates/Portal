# coding: utf-8
import struct, math, gzip, os
import numpy
from osgeo import gdal
from gdalconst import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
import histeq, shading
%matplotlib inline

#fig = plt.figure(figsize=(128, 64),dpi=144,frameon=False)
fig = plt.figure(figsize=(12, 6),dpi=144,frameon=False)
m = Basemap(
    projection='cyl',
    llcrnrlat=-90,
    urcrnrlat=90,
    llcrnrlon=-180,
    urcrnrlon=180,
    resolution=None)
m.drawmapboundary(linewidth=0)
dataset = gdal.Open('/mnt/workspace/EMAG2/EMAG2_2m_nearest.nc', GA_ReadOnly )
band = dataset.GetRasterBand(1)
r = band.ReadAsArray( 0, 0, band.XSize, band.YSize, 7600, 3800)
#r = band.ReadAsArray( 0, 0, band.XSize, band.YSize, band.XSize, band.YSize)

#v_min = numpy.nanmin(r)
#v_max = numpy.nanmax(r)

colors = [
       
        [30,30, 30], # -200
        [33,40,230 ],    
        [62,104,230],      
        [90,168,230],        
        [150,200,230],    
        [255,255,255],      
        [255,255,255],       
        [230,230,157],       
        [230,230,30],      
        [230,162,60],        
        [230,40,91],      
        [168,30,168],# 200
        
    ]

steps = [
    -200,       
    -60,     
    -48,  
    -36,    
    -20,    
    -4,   
    4,   
    20,     
    36,   
    48,   
    60, 
    200,
    ]


color_list = []
for i in range(len(steps)):
    color_list.append((float(steps[i]-steps[0])/(steps[-1]-steps[0]), [x/255.0 for x in colors[i]]))
    


my_cmap = LinearSegmentedColormap.from_list('my_cmap', color_list, N=1024)


where_are_nan =numpy.isnan(r)

r[where_are_nan] =0
where_less_than_minus_200 = (r<-200)
r[where_less_than_minus_200] = -200
where_greater_than_200 = (r>200)
r[where_greater_than_200] = 200

rgb = shading.shade(r,shading.intensity(r), cmap=my_cmap)

rgb[where_are_nan] = [128.0/255,128.0/255,128.0/255]

#masked_data = numpy.ma.masked_where(numpy.isnan(rgb),rgb)
#print masked_data
#m.imshow(masked_data,cmap=cmap,interpolation='sinc')
rgb = numpy.flipud(rgb)
m.imshow(rgb,interpolation='sinc')

#plot.show()
fig.savefig('/mnt/workspace/EMAG2/EMAG2.tiff',bbox_inches='tight',pad_inches=0,dpi=144,transparent=True,frameon=False)