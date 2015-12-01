#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    Filename: split_grid.py
    Author: Xiaodong Qin
    Email: michael.chin@sydney.edu.au
    
    Some grid files are very large, e.g. the srtm15_plus grid is more than 15GB.
    Because of the limit of computer architecture and hardware, sometimes, it is not feasible
    to process a file so large.
    The solution is to divide and conquer.  
    This Python function splits a large grid file into smaller pieces so that they can be processed separately.
    Later on, the small pieces are bound together again to create a virtual raster with 'gdalbuildvrt'.
    The gdal_translate command is called to do the actural cutting. The gdal_translate command must be in the system path.
    
    Parameters:
        input_file: the full path of the grid file to be split
                    the grid file must be georefereced and have a global extent.
        output_dir: the full path of a folder where the pieces of this grid file will be stored 
                    If the folder does not exist, a new folder will be created. So make sure the program
                    has the 'write' permission.
        factor:     the factor parameter determines the size of the pieces.The input file will be split into
                    factor*factor pieces. The default value is 10 which will split the grid file into 100 pieces evenly.
    Return:        
            0 if the function is successful
            -1 if error occurs
'''
import os

def split_grid(input_file, output_dir, factor=10):
    lon_inc = 360/factor
    lat_inc = 180/factor

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(factor):
        for j in range(factor):
            ulx=-180+i*lon_inc
            uly=90-j*lat_inc
            lrx=-180+(i+1)*lon_inc
            lry=90-(j+1)*lat_inc
            cmd = "gdal_translate -projwin {0} {1} {2} {3} {4} {5}{6}_{7}.tif".format(
                ulx,uly,lrx,lry,input_file,output_dir,i,j)
            print cmd
            ret = os.system(cmd)
            if ret != 0:
                return -1
    return 0