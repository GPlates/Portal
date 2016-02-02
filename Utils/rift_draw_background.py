#Draw background
#gdal2tiles.py --profile=geodetic --zoom='0-4' -s EPSG:4326 --no-kml rift_ref.tiff
#gdal_translate -a_ullr -180 90 180 -90 rift.tiff rift_ref.tiff

from mpl_toolkits.basemap import Basemap, cm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import os
%matplotlib inline
SHPT_POINT      =  1   #Points
SHPT_ARC        =  3   #Arcs (Polylines, possible in parts)
SHPT_POLYGON    =  5   #Polygons (possible in parts)
SHPT_MULTIPOINT =  8   #MultiPoint (related points)

root_dir = '/mnt/workspace/rift/background/'

for i in range(180,241):
    m = Basemap(
        projection='cyl',
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
        resolution=None)
    m.drawmapboundary(linewidth=0)

    s = m.readshapefile('{0}reconstructed_{1}Ma/reconstructed_{1}Ma_polygon'.format(root_dir, i),
                        'shpfile', color='grey', drawbounds=True, linewidth=0.5)
    if s[1] == SHPT_POLYGON :
        for xy in m.shpfile:
            poly = Polygon(xy, edgecolor='none', facecolor='grey', alpha=0.4)
            plt.gca().add_patch(poly)

    s = m.readshapefile('{0}reconstructed_{1}Ma/reconstructed_{1}Ma_point'.format(root_dir, i),'shpfile')
    xy = [list(t) for t in zip(*m.shpfile)]
    m.scatter(xy[0],xy[1],c='grey', s=10, edgecolor='', alpha=1.0,marker='.')
    
    s = m.readshapefile('{0}reconstructed_{1}Ma/reconstructed_{1}Ma_polyline'.format(root_dir, i),
                        'shpfile', color='grey', drawbounds=True, linewidth=.5)

    fig = plt.gcf()
    fig.set_size_inches(16,8)
    fig.patch.set_facecolor('lightgrey')
    
    #plt.show()
    fig.savefig('/mnt/tmp/rift/{0}.tiff'.format(i),bbox_inches='tight',facecolor='lightgrey',pad_inches=0,dpi=720,transparent=True,frameon=False)
    os.system('gdal_translate -a_ullr -180 90 180 -90 /mnt/tmp/rift/{0}.tiff /mnt/tmp/rift/{0}_ref.tiff'.format(i))
    os.system('gdal2tiles.py --profile=geodetic --zoom="0-4" -s EPSG:4326 --no-kml /mnt/tmp/rift/{0}_ref.tiff /mnt/tmp/rift/{0}'.format(i))
    plt.clf()
    print i
print 'done'