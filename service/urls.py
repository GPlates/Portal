from django.conf.urls import patterns, include, url
from . import apis, views

urlpatterns =[
    url(r'^$', views.index, name='index'),
    url(r'^api_ref/$', views.api_ref, name='api_reference'),
    url(r'^examples/$', views.api_examples, name='api_examples'),
    url(r'^reconstruct_points/$', apis.reconstruct_points, name='reconstruct_points'),
    url(r'^get_coastline_polygons/$', apis.get_coastline_polygons, name='get_coastline_polygons'),
    url(r'^reconstruct_feature_collection/', apis.reconstruct_feature_collection, name='reconstruct_feature_collection'),
]
