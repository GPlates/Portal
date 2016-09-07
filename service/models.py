# -*- coding: utf-8 -*-
from django.db import models
from django.contrib.gis.db import models as geo_models

class CoastlinePolygons(geo_models.Model):
    plateid1 = geo_models.DecimalField(max_digits=10, decimal_places=0, blank=True, null=True)
    fromage = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    toage = models.DecimalField(max_digits=65535, decimal_places=65535, blank=True, null=True)
    name = models.CharField(max_length=80, blank=True, null=True)
    geom = geo_models.MultiPolygonField(srid=4269, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'coastline_polygons'
