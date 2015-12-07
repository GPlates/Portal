#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
	The split_grid.py splits a large grid file into smaller pieces.
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
	
    Filename: split_grid.py
    Author: Xiaodong Qin
    Email: michael.chin@sydney.edu.au
    
	Get latest version and submit bug report at 
	GitHub: https://github.com/GPlates/Portal/blob/master/Utils/split_grid.py
	
    Some grid files are very large, e.g. the srtm15_plus grid is more than 15GB.
    Because of the limit of computer architecture and hardware, sometimes, it is not feasible
    to process a file so large.
    The solution is to divide and conquer.  
    This Python function splits a large grid file into smaller pieces so that they can be processed separately.
    Later on, the small pieces are bound together again to create a virtual raster with 'gdalbuildvrt'.
    The gdal_translate command, which must be in the system path, is called to do the actural cutting. 
    
    Parameters:
        input_file: the full path of the grid file to be split
                    the grid file must be georefereced and have a global extent.
        output_dir: the full path of a folder where the pieces of this grid file will be stored 
                    If the folder does not exist, a new folder will be created. So make sure the program
                    has the 'write' permission.
        factor:     the factor parameter determines the size of the pieces.The input file will be split into
                    factor*factor pieces. The default value is 10 which will split the grid file into 100 pieces evenly and
					each piece will be georeferenced.
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