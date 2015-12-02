#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    Filename: terrain_tiles.py
    Author: Xiaodong Qin
    Email: michael.chin@sydney.edu.au
    
	Get latest version and submit bug report at 
	GitHub: https://github.com/GPlates/Portal/blob/master/Utils/terrain_tiles.py
	
    The 'create_tiles" function generates terrain tiles from an elevation dataset. 
    
    Parameters:
        input_file: the full path of the elevation dataset
                    
        output_dir: the full path of a folder where the terrain tiles will be stored 
                    
        max_level:  the max level of terrain tiles
    Return:        
            this function does not return any data
'''
import struct, math, gzip, os
import numpy
from osgeo import gdal

TILE_SIZE = 64
tail = struct.pack('BB', 15, 0)
tail_last = struct.pack('BB', 0, 0)

#internal function
def cut_tiles(band, level):
    x = 2**(level+1)
    y = 2**level
    
    img = band.ReadAsArray( 0, 0, band.XSize, band.YSize, TILE_SIZE*x, TILE_SIZE*y ).astype(numpy.short) 
    new_img = numpy.zeros((TILE_SIZE*y+1,TILE_SIZE*x+1), dtype=numpy.short)
    new_img[:-1,:-1] = img
    new_img[-1,:-1] = img[0]
    new_img[:,-1] = new_img[:,0]
              
    ret = numpy.zeros((x,y,TILE_SIZE+1,TILE_SIZE+1), dtype=numpy.short)
    for j in range(y):
        for i in range(x):
            ret[i][j] = (new_img[TILE_SIZE*(y-j-1):TILE_SIZE*(y-j)+1, TILE_SIZE*i:TILE_SIZE*(i+1)+1])
    return ret


def create_tiles(input_file, output_dir, max_level=7):
    dataset = gdal.Open(input_file, GA_ReadOnly )
    band = dataset.GetRasterBand(1)

    for level in range(max_level+1):
        tiles = cut_tiles(band, level)
        print tiles.shape
        for i in range(tiles.shape[0]):
            for j in range(tiles.shape[1]):
                filename=output_dir+'/{0}/{1}/{2}.terrain'.format(level,i,j)
                if not os.path.exists(os.path.dirname(filename)):
                    os.makedirs(os.path.dirname(filename))
                f = gzip.open(filename,'wb')
                if level == maxlevel:
                    f.write(tiles[i][j].tostring()+tail_last)
                else:
                    f.write(tiles[i][j].tostring()+tail)
                f.close()