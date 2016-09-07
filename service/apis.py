# -*- coding: utf-8 -*-
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, HttpResponseServerError, HttpResponseNotAllowed
from django.conf import settings

import sys, json

import pygplates

from models import CoastlinePolygons
from portal.models import StaticPolygons

Model_Root = settings.HOME_DIR+'/Models/'

MODEL_DEFAULT = Model_Root+'default/'

def reconstruct_points(request):
    points = request.GET.get('points', None)
    plate_id = request.GET.get('pid', None)
    time = request.GET.get('time', None)
    
    rotation_model = pygplates.RotationModel(
        MODEL_DEFAULT+"Seton_etal_ESR2012_2012.1.rot")
    static_polygons_filename = \
        MODEL_DEFAULT+"Seton_etal_ESR2012_StaticPolygons_2012.1.gpmlz"
    
    point_features = []
    if points:
        ps = points.split(',')
        if len(ps)%2==0:
            for lat,lon in zip(ps[1::2], ps[0::2]):
                point_feature = pygplates.Feature()
                point_feature.set_geometry(pygplates.PointOnSphere(float(lat),float(lon)))    
                point_features.append(point_feature)

    #for f in point_features:
    #    f.set_reconstruction_plate_id(int(plate_id))
    assigned_point_features = pygplates.partition_into_plates(
        static_polygons_filename,
        rotation_model,
        point_features,
        properties_to_copy = [
            pygplates.PartitionProperty.reconstruction_plate_id,
            pygplates.PartitionProperty.valid_time_period])
    assigned_point_feature_collection = pygplates.FeatureCollection(assigned_point_features)
    reconstructed_feature_geometries = []
    pygplates.reconstruct(assigned_point_feature_collection, rotation_model, reconstructed_feature_geometries, float(time))
    ret='{"coordinates":['
    for g in reconstructed_feature_geometries:
        ret+='[{0:5.2f},{1:5.2f}],'.format(
            g.get_reconstructed_geometry().to_lat_lon()[1],
            g.get_reconstructed_geometry().to_lat_lon()[0])
    ret=ret[0:-1]
    ret+=']}'
    return HttpResponse(ret, content_type='application/json')


class PrettyFloat(float):
    def __repr__(self):
        return '%.2f' % self

def pretty_floats(obj):
    if isinstance(obj, float):
        return PrettyFloat(obj)
    elif isinstance(obj, dict):
        return dict((k, pretty_floats(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return map(pretty_floats, obj)             
    return obj

import cProfile , pstats, ast, StringIO

def get_coastline_polygons(request):
    #pr = cProfile.Profile()
    #pr.enable()
    
    time = request.GET.get('time', 0)
    features = []
    ''' 
    polygons = CoastlinePolygons.objects.all()
    #polygons = StaticPolygons.objects.all()

    for p in polygons:
        polygon_feature = pygplates.Feature()
        polygon_feature.set_geometry(
            pygplates.PolygonOnSphere([(lat,lon) for lon, lat in p.geom[0][0]]))
        polygon_feature.set_reconstruction_plate_id(int(p.plateid1))
        features.append(polygon_feature)
    '''
    import shapefile 
    sf = shapefile.Reader(
        MODEL_DEFAULT+"coastlines_low_res/Seton_etal_ESR2012_Coastlines_2012.shp") 
    for record in sf.shapeRecords():
        if record.shape.shapeType != 5:
            continue
        for idx in range(len(record.shape.parts)):
            start_idx = record.shape.parts[idx]
            end_idx = len(record.shape.points)
            if idx < (len(record.shape.parts) -1):
                end_idx = record.shape.parts[idx+1]
            polygon_feature = pygplates.Feature()
            points = record.shape.points[start_idx:end_idx]
            polygon_feature.set_geometry(
                pygplates.PolygonOnSphere([(lat,lon) for lon, lat in points]))
            polygon_feature.set_reconstruction_plate_id(int(record.record[0]))
            features.append(polygon_feature)
            break 
     
    feature_collection = pygplates.FeatureCollection(features)
    reconstructed_polygons = []
    rotation_model = pygplates.RotationModel(
        MODEL_DEFAULT+"Seton_etal_ESR2012_2012.1.rot")    

    '''
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(20)
    print s.getvalue()
    '''
    
    pygplates.reconstruct(
        feature_collection, 
        rotation_model, 
        reconstructed_polygons,
        float(time))
    
    #return HttpResponse('OK')
    data = {"type": "FeatureCollection"}
    data["features"] = [] 
    for p in reconstructed_polygons:
        feature = {"type": "Feature"}
        feature["geometry"] = {}
        feature["geometry"]["type"] = "Polygon"
        feature["geometry"]["coordinates"] = [[(lon,lat) for lat, lon in p.get_reconstructed_geometry().to_lat_lon_list()]]
        data["features"].append(feature)
    ret = json.dumps(pretty_floats(data))
   
    return HttpResponse(ret, content_type='application/json')


def reconstruct_feature_collection(request):
    DATA_DIR = Model_Root+'caltech/'

    if request.method == 'POST':
        return HttpResponse('POST method is not accepted for now.')

    geologicage = request.GET.get('geologicage', 140)
    output_format = request.GET.get('output', 'geojson')
    fc_str = request.GET.get('feature_collection')
    fc = json.loads(fc_str)
 
    features=[]
    for f in fc['features']:
        geom = f['geometry']
        feature = pygplates.Feature()
        if geom['type'] == 'Point':
            feature.set_geometry(pygplates.PointOnSphere(
                float(geom['coordinates'][1]),
                float(geom['coordinates'][0])))
        if geom['type'] == 'LineString':
            feature.set_geometry(
                pygplates.PolylineOnSphere([(point[1],point[0]) for point in geom['coordinates']]))
        if geom['type'] == 'Polygon':
            feature.set_geometry(
                pygplates.PolygonOnSphere([(point[1],point[0]) for point in geom['coordinates'][0]]))
        if geom['type'] == 'MultiPoint':
             feature.set_geometry(
                pygplates.MultiPointOnSphere([(point[1],point[0]) for point in geom['coordinates']]))

        features.append(feature)


    if float(geologicage) < 250:
        rotation_files = [DATA_DIR+'/Seton_etal_ESR2012_2012.1.rot']
    else :
        rotation_files = [DATA_DIR+'/Global_EB_410-250Ma_GK07_Matthews_etal.rot']

    rotation_model = pygplates.RotationModel(rotation_files)

    assigned_features = pygplates.partition_into_plates(
        DATA_DIR+'Seton_etal_ESR2012_StaticPolygons_2012.1.gpmlz',
        rotation_model,
        features,
        properties_to_copy = [
            pygplates.PartitionProperty.reconstruction_plate_id,
            pygplates.PartitionProperty.valid_time_period],
        partition_method = pygplates.PartitionMethod.most_overlapping_plate
    )


    reconstructed_geometries = []
    pygplates.reconstruct(assigned_features, rotation_model, reconstructed_geometries, float(geologicage), 0)
    
    
    data = {"type": "FeatureCollection"}
    data["features"] = []
    for g in reconstructed_geometries:
        geom =  g.get_reconstructed_geometry()
        feature = {"type": "Feature"}
        feature["geometry"] = {}
        if isinstance(geom, pygplates.PointOnSphere):
            feature["geometry"]["type"] = "Point"
            p = geom.to_lat_lon_list()[0]
            feature["geometry"]["coordinates"] = [p[1], p[0]]
        elif isinstance(geom, pygplates.MultiPointOnSphere):
            feature["geometry"]["type"] = 'MultiPoint'
            feature["geometry"]["coordinates"] = [[lon,lat] for lat, lon in geom.to_lat_lon_list()]
        elif isinstance(geom, pygplates.PolylineOnSphere):
            feature["geometry"]["type"] = 'LineString'
            feature["geometry"]["coordinates"] = [[lon,lat] for lat, lon in geom.to_lat_lon_list()]
        elif isinstance(geom, pygplates.PolygonOnSphere):
            feature["geometry"]["type"] = 'Polygon'
            feature["geometry"]["coordinates"] = [[[lon,lat] for lat, lon in geom.to_lat_lon_list()]]
        else:
            raise 'Unrecognized Geometry Type.'
        
        feature["properties"]={}    
        
        data["features"].append(feature)

    ret = json.dumps(pretty_floats(data))

    return HttpResponse(ret, content_type='application/json')

